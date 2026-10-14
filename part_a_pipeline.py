from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
DB = ROOT / "data" / "instamart.db"
OUTPUTS = ROOT / "outputs"
CITY_MAP = {"bangalore": "Bengaluru", "bengaluru": "Bengaluru", "bombay": "Mumbai", "delhi": "Delhi", "chennai": "Chennai", "hyderabad": "Hyderabad", "pune": "Pune", "mumbai": "Mumbai"}
GENDER_MAP = {"m": "Male", "male": "Male", "f": "Female", "female": "Female"}
ORDER_FIELDS = ["order_id", "customer_id", "product_id", "quantity", "order_ts", "status", "discount_pct", "order_amount", "delivery_minutes"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def clean_customers() -> list[dict[str, object]]:
    rows = read_csv(RAW / "customers.csv")
    ages = []
    for row in rows:
        try:
            age = int(row["age"])
        except (TypeError, ValueError):
            age = None
        row["_age"] = age if age is not None and 18 <= age <= 100 else None
        if row["_age"] is not None:
            ages.append(row["_age"])
    imputed_age = int(round(median(ages)))
    result = []
    for row in rows:
        city_key = row["city"].strip().lower()
        gender_key = row["gender"].strip().lower()
        try:
            signup = datetime.strptime(row["signup_date"], "%Y-%m-%d").date().isoformat()
        except ValueError:
            continue
        result.append({
            "customer_id": row["customer_id"].strip(),
            "city": CITY_MAP.get(city_key, city_key.title()),
            "signup_date": signup,
            "age": row["_age"] if row["_age"] is not None else imputed_age,
            "gender": GENDER_MAP.get(gender_key, "Unknown"),
        })
    return result


def clean_products() -> list[dict[str, object]]:
    rows = read_csv(RAW / "products.csv")
    seen = set()
    clean = []
    for row in rows:
        product_id = row["product_id"].strip()
        try:
            price = float(row["price_inr"])
        except (TypeError, ValueError):
            continue
        if not product_id or product_id in seen or price <= 0:
            continue
        seen.add(product_id)
        clean.append({
            "product_id": product_id,
            "product_name": row["product_name"].strip(),
            "category": row["category"].strip(),
            "price_inr": round(price, 2),
        })
    return clean


def clean_orders() -> tuple[list[dict[str, object]], dict[str, int]]:
    raw = read_csv(RAW / "orders.csv")
    product_prices = {row["product_id"]: float(row["price_inr"]) for row in read_csv(RAW / "products.csv")}
    selected: dict[str, dict[str, str]] = {}
    rank = {"pending": 0, "cancelled": 1, "completed": 2}
    for row in raw:
        order_id = row["order_id"].strip()
        try:
            timestamp = datetime.strptime(row["order_ts"], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            timestamp = datetime.min
        score = (rank.get(row["status"].strip().lower(), -1), timestamp)
        previous = selected.get(order_id)
        if previous is None:
            selected[order_id] = row
            continue
        try:
            previous_timestamp = datetime.strptime(previous["order_ts"], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            previous_timestamp = datetime.min
        previous_score = (rank.get(previous["status"].strip().lower(), -1), previous_timestamp)
        if score > previous_score:
            selected[order_id] = row

    clean = []
    amount_imputed = 0
    delivery_set_missing = 0
    for row in selected.values():
        status = row["status"].strip().lower()
        try:
            quantity = int(row["quantity"])
            discount = float(row["discount_pct"])
            timestamp = datetime.strptime(row["order_ts"], "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S")
        except (TypeError, ValueError):
            continue
        try:
            amount = float(row["order_amount"])
        except (TypeError, ValueError):
            price = product_prices.get(row["product_id"])
            if price is None:
                continue
            amount = round(price * quantity * (1 - discount / 100), 2)
            amount_imputed += 1
        try:
            delivery = int(row["delivery_minutes"]) if row["delivery_minutes"].strip() else None
        except ValueError:
            delivery = None
        if delivery is not None and not 8 <= delivery <= 75:
            delivery = None
        if status == "completed" and delivery is None:
            delivery_set_missing += 1
        valid = (
            status in {"completed", "cancelled"}
            and 1 <= quantity <= 20
            and 0 <= discount <= 100
            and amount > 0
        )
        if not valid:
            continue
        clean.append({
            "order_id": row["order_id"].strip(),
            "customer_id": row["customer_id"].strip(),
            "product_id": row["product_id"].strip(),
            "quantity": quantity,
            "order_ts": timestamp,
            "status": status,
            "discount_pct": round(discount, 2),
            "order_amount": round(amount, 2),
            "delivery_minutes": delivery if delivery is not None else "",
        })
    counts = {
        "raw_rows": len(raw),
        "duplicate_rows_removed": len(raw) - len(selected),
        "invalid_rows_removed": len(selected) - len(clean),
        "order_amounts_imputed": amount_imputed,
        "completed_delivery_values_set_missing": delivery_set_missing,
        "clean_rows": len(clean),
        "completed_rows": sum(row["status"] == "completed" for row in clean),
        "cancelled_rows": sum(row["status"] == "cancelled" for row in clean),
    }
    return clean, counts


def run() -> dict[str, int]:
    CLEAN.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    customers = clean_customers()
    products = clean_products()
    orders, quality = clean_orders()
    quality["clean_customers"] = len(customers)
    quality["clean_products"] = len(products)
    write_csv(CLEAN / "customers.csv", ["customer_id", "city", "signup_date", "age", "gender"], customers)
    write_csv(CLEAN / "products.csv", ["product_id", "product_name", "category", "price_inr"], products)
    write_csv(CLEAN / "orders.csv", ORDER_FIELDS, orders)
    with sqlite3.connect(DB) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript("""
            DROP TABLE IF EXISTS orders;
            DROP TABLE IF EXISTS products;
            DROP TABLE IF EXISTS customers;
            CREATE TABLE customers (
                customer_id TEXT PRIMARY KEY, city TEXT NOT NULL, signup_date TEXT NOT NULL,
                age INTEGER NOT NULL, gender TEXT NOT NULL
            );
            CREATE TABLE products (
                product_id TEXT PRIMARY KEY, product_name TEXT NOT NULL, category TEXT NOT NULL,
                price_inr REAL NOT NULL CHECK (price_inr > 0)
            );
            CREATE TABLE orders (
                order_id TEXT PRIMARY KEY, customer_id TEXT NOT NULL, product_id TEXT NOT NULL,
                quantity INTEGER NOT NULL CHECK (quantity > 0), order_ts TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('completed', 'cancelled')),
                discount_pct REAL NOT NULL CHECK (discount_pct BETWEEN 0 AND 100),
                order_amount REAL NOT NULL CHECK (order_amount > 0), delivery_minutes INTEGER,
                FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
                FOREIGN KEY (product_id) REFERENCES products(product_id)
            );
        """)
        connection.executemany("INSERT INTO customers VALUES (?, ?, ?, ?, ?)", [tuple(row.values()) for row in customers])
        connection.executemany("INSERT INTO products VALUES (?, ?, ?, ?)", [tuple(row.values()) for row in products])
        connection.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", [tuple(row.values()) for row in orders])
        connection.execute("CREATE INDEX idx_orders_customer_date ON orders(customer_id, order_ts)")
        connection.execute("CREATE INDEX idx_orders_product ON orders(product_id)")
    (OUTPUTS / "part_a_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    return quality


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
