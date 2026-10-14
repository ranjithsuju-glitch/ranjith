-- Trusted analytical queries for the Part A SQLite database.
SELECT city, COUNT(*) AS customers
FROM customers
GROUP BY city
ORDER BY customers DESC, city;

SELECT status, COUNT(*) AS orders, ROUND(SUM(order_amount), 2) AS order_value_inr
FROM orders
GROUP BY status
ORDER BY status;

SELECT p.category, COUNT(*) AS completed_orders,
       ROUND(SUM(o.order_amount), 2) AS revenue_inr,
       ROUND(AVG(o.delivery_minutes), 2) AS mean_delivery_minutes
FROM orders AS o
JOIN products AS p USING (product_id)
WHERE o.status = 'completed'
GROUP BY p.category
ORDER BY revenue_inr DESC;

SELECT strftime('%Y-%m', order_ts) AS order_month,
       COUNT(*) AS completed_orders,
       ROUND(SUM(order_amount), 2) AS revenue_inr
FROM orders
WHERE status = 'completed'
GROUP BY order_month
ORDER BY order_month;
