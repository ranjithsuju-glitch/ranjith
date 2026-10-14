from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix

import mock_llm

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
MAX_INPUT_TOKENS = 400
TOKEN_BUDGET = 6000
BASE_DELAY = 0.05
ALLOWED = ["billing", "delivery", "technical", "fraud"]
PATTERNS = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
    ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]"),
]
ZERO_SHOT_PROMPT = """CLASSIFY each support ticket into exactly one allowed category: billing, delivery, technical, fraud.
Reply with JSON only in this form: {"category": "...", "reason": "..."}."""
FEW_SHOT_PROMPT = """CLASSIFY each support ticket into exactly one allowed category: billing, delivery, technical, fraud.
Reply with JSON only in this form: {"category": "...", "reason": "..."}.
EXAMPLES:
Ticket: I see an unauthorized card transaction. Category: fraud.
Ticket: My order arrived late. Category: delivery."""


def redact(text: str) -> tuple[str, dict[str, int]]:
    redacted = text
    counts = {}
    for name, pattern, placeholder in PATTERNS:
        redacted, counts[name] = pattern.subn(placeholder, redacted)
    assert all(pattern.search(redacted) is None for _, pattern, _ in PATTERNS)
    return redacted, counts


def retry_chat(ticket_id: str, messages: list[dict[str, str]], rng: random.Random):
    retries = 0
    while True:
        try:
            return mock_llm.chat(messages), retries
        except mock_llm.TransientLLMError:
            if retries >= 3:
                raise
            wait = BASE_DELAY * (2 ** retries) * rng.uniform(0.9, 1.15)
            print(f"retry ticket_id={ticket_id} retry_number={retries} wait={wait:.4f}s")
            time.sleep(wait)
            retries += 1


def validate_response(text: str) -> tuple[str, str]:
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        words = re.findall(r"[a-z]+", (text or "").lower())
        matches = [category for category in ALLOWED if category in words]
        if len(matches) == 1:
            return matches[0], "classified_keyword"
        return "", "flagged_invalid_output"
    category = payload.get("category") if isinstance(payload, dict) else None
    if category in ALLOWED:
        return category, "classified_json"
    return "", "flagged_invalid_output"


def _base_row(ticket, status: str, category: str = "", tokens: int = 0) -> dict[str, object]:
    return {
        "ticket_id": ticket.ticket_id,
        "customer_id": ticket.customer_id,
        "status": status,
        "category": category,
        "true_category": ticket.true_category,
        "tokens_used": tokens,
    }


def run_classification(prompt_name: str, token_budget: int | None = None) -> tuple[pd.DataFrame, dict[str, object]]:
    tickets = pd.read_csv(RAW / "tickets.csv", keep_default_na=False)
    prompt = ZERO_SHOT_PROMPT if prompt_name == "zero_shot" else FEW_SHOT_PROMPT
    mock_llm.reset()
    rng = random.Random(42)
    running_tokens = 0
    rows = []
    calls = 0
    for ticket in tickets.itertuples(index=False):
        text = ticket.text
        if not text.strip():
            rows.append(_base_row(ticket, "skipped_empty"))
            continue
        estimated = mock_llm.est_tokens(text)
        if estimated > MAX_INPUT_TOKENS:
            rows.append(_base_row(ticket, "flagged_long"))
            continue
        if token_budget is not None and running_tokens >= token_budget:
            rows.append(_base_row(ticket, "budget_exhausted"))
            continue
        redacted, _ = redact(text)
        assert redacted == text or all(pattern.search(redacted) is None for _, pattern, _ in PATTERNS)
        assert all(pattern.search(redacted) is None for _, pattern, _ in PATTERNS)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": redacted},
        ]
        assert messages[-1]["content"] == redacted
        assert all(pattern.search(messages[-1]["content"]) is None for _, pattern, _ in PATTERNS)
        response, retries = retry_chat(ticket.ticket_id, messages, rng)
        calls += 1
        used = int(response.usage.total_tokens)
        running_tokens += used
        category, status = validate_response(response.choices[0].message.content)
        row = _base_row(ticket, status, category, used)
        row["retries"] = retries
        rows.append(row)

    frame = pd.DataFrame(rows)
    classified = frame[frame["status"].isin(["classified_json", "classified_keyword"])]
    accuracy = float(classified["true_category"].eq(classified["category"]).mean()) if len(classified) else 0.0
    stats = {
        "prompt": prompt_name,
        "tickets": int(len(frame)),
        "classified": int(len(classified)),
        "accuracy_classified": accuracy,
        "total_tokens": int(running_tokens),
        "tokens_per_classified_ticket": float(running_tokens / len(classified)) if len(classified) else 0.0,
        "model_calls": calls,
        "retries": int(frame.get("retries", pd.Series(dtype=int)).sum()),
        "status_counts": {str(status): int(count) for status, count in frame["status"].value_counts().items()},
        "token_budget": token_budget,
    }
    return frame, stats


def privacy_audit() -> dict[str, int]:
    tickets = pd.read_csv(RAW / "tickets.csv", keep_default_na=False)
    counts = {name: 0 for name, _, _ in PATTERNS}
    residual = 0
    for text in tickets["text"]:
        redacted, found = redact(text)
        for name in counts:
            counts[name] += found[name]
        residual += sum(len(pattern.findall(redacted)) for _, pattern, _ in PATTERNS)
    counts["total_substitutions"] = sum(counts.values())
    counts["remaining_matches"] = residual
    counts["tickets"] = len(tickets)
    assert residual == 0
    return counts


def route_final(frame: pd.DataFrame) -> pd.DataFrame:
    segments = pd.read_csv(OUT / "customer_segments.csv", keep_default_na=False)
    premium = dict(zip(segments["customer_id"], segments["is_premium"].astype(str).str.lower().eq("true")))
    segment_names = dict(zip(segments["customer_id"], segments["segment_name"]))
    result = frame.copy()
    result["segment_name"] = result["customer_id"].map(segment_names).fillna("")
    result["is_premium"] = result["customer_id"].map(premium).fillna(False)
    result["priority"] = "NORMAL"
    result["route"] = "support_queue"
    needs_human = result["status"].isin([
        "skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted",
    ])
    result.loc[needs_human, "route"] = "human_review"
    fraud = ~needs_human & result["category"].eq("fraud")
    result.loc[fraud, "route"] = "security_team"
    result.loc[fraud, "priority"] = "HIGH"
    premium = ~needs_human & ~fraud & result["is_premium"]
    result.loc[premium, "route"] = "premium_desk"
    result.loc[premium, "priority"] = "HIGH"
    return result[["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]]


def run() -> dict[str, object]:
    OUT.mkdir(parents=True, exist_ok=True)
    privacy = privacy_audit()
    zero, zero_stats = run_classification("zero_shot")
    few, few_stats = run_classification("few_shot")
    budget, budget_stats = run_classification("zero_shot", TOKEN_BUDGET)
    final_few, final_few_stats = run_classification("few_shot", 100000)
    zero.to_csv(OUT / "triage_zero_shot.csv", index=False)
    few.to_csv(OUT / "triage_few_shot.csv", index=False)
    budget.to_csv(OUT / "triage_budget_demo.csv", index=False)
    classified_zero = zero[zero["status"].isin(["classified_json", "classified_keyword"])]
    confusion = pd.crosstab(
        pd.Categorical(classified_zero["true_category"], categories=ALLOWED),
        pd.Categorical(classified_zero["category"], categories=ALLOWED),
        rownames=["true_category"], colnames=["predicted_category"], dropna=False,
    )
    confusion.to_csv(OUT / "triage_zero_shot_confusion.csv")
    final_frame = route_final(final_few)
    final_frame.to_csv(OUT / "triage_results.csv", index=False)
    report = {
        "privacy": privacy,
        "zero_shot": zero_stats,
        "few_shot": few_stats,
        "final_few_shot": final_few_stats,
        "budget_demo": budget_stats,
        "budget_exhausted_count": int(budget["status"].eq("budget_exhausted").sum()),
        "final_status_counts": {str(status): int(count) for status, count in final_frame["status"].value_counts().items()},
        "final_route_counts": {str(route): int(count) for route, count in final_frame["route"].value_counts().items()},
    }
    (OUT / "part_c_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
