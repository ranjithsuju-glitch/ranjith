from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


def load_json(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def csv_rows(name: str) -> list[dict[str, str]]:
    with (OUT / name).open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(cell).replace("|", "\\|").replace("\n", " ") for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def number(value, digits: int = 3) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_readme() -> str:
    quality = load_json("part_a_quality.json")
    b = load_json("part_b_summary.json")
    b_rfm, b_model = b["rfm"], b["lapse"]
    c = load_json("part_c_report.json")
    d = load_json("part_d_report.json")

    part_a_table = table(
        ["Measure", "Result"],
        [[key.replace("_", " ").title(), value] for key, value in quality.items()],
    )
    k_table = table(
        [[row["k"], number(row["inertia"], 1), number(row["silhouette"], 3)] for row in b_rfm["kmeans"]],
    )
    profiles = csv_rows("segment_profile.csv")
    profile_table = table(
        ["Segment ID", "Name", "Size", "Mean recency", "Mean frequency", "Mean monetary (INR)", "Recommended action", "Premium"],
        [[row["segment_id"], row["segment_name"], row["size"], number(float(row["mean_recency"]), 2),
          number(float(row["mean_frequency"]), 2), number(float(row["mean_monetary"]), 2),
          row["recommended_action"], row["is_premium"]] for row in profiles],
    )
    pca = b_rfm["pca"]["explained_variance_ratio"]
    lapse_rows = []
    for name, result in (("Always not lapsed", b_model["baseline"]),
                         ("Logistic regression", b_model["logistic"]["test"]),
                         ("Random forest", b_model["random_forest"]["test"])):
        lapse_rows.append([name, json.dumps(result["confusion_matrix"]), number(result.get("precision", 0)),
                           number(result.get("recall", 0)), number(result.get("f1", 0)),
                           number(result.get("accuracy", 0)) if "accuracy" in result else "-"])
    model_table = table(["Model", "Confusion matrix [[TN, FP], [FN, TP]]", "Lapsed precision", "Lapsed recall", "Lapsed F1", "Accuracy"], lapse_rows)
    grid_table = table(
        ["Model", "Best parameters", "Best CV F1"],
        [["Logistic regression", json.dumps(b_model["logistic"]["best_params"], sort_keys=True), number(b_model["logistic"]["best_cv_f1"])],
         ["Random forest", json.dumps(b_model["random_forest"]["best_params"], sort_keys=True), number(b_model["random_forest"]["best_cv_f1"])]],
    )
    balanced = b_model["balanced_logistic_test"]
    importance_table = table(["Feature", "Importance"], [[row["feature"], number(row["importance"])] for row in b_model["random_forest_top_features"]])
    coefficient_readings = {
        "recency": {"positive": "Longer time since the last order is associated with higher lapse odds.", "negative": "Longer time since the last order is associated with lower lapse odds in this fitted model."},
        "frequency": {"positive": "More completed orders are associated with higher lapse odds in this fitted model.", "negative": "More completed orders are associated with lower lapse odds."},
        "monetary": {"positive": "Higher historical spend is associated with higher lapse odds in this fitted model.", "negative": "Higher historical spend is associated with lower lapse odds."},
    }
    coefficient_table = table(["Feature", "Coefficient", "Direction", "Business reading"], [
        [feature, number(values["coefficient"]), values["sign"], coefficient_readings[feature][values["sign"]]]
        for feature, values in b_model["logistic_coefficients"].items()
    ])
    retention = csv_rows("retention_list.csv")

    privacy = c["privacy"]
    c_metrics = []
    for label in ("zero_shot", "few_shot"):
        item = c[label]
        c_metrics.append([label.replace("_", " ").title(), item["classified"],
                          f"{100 * item['accuracy_classified']:.1f}%", item["total_tokens"],
                          number(item["tokens_per_classified_ticket"], 1), item["retries"]])
    c_table = table(["Run", "Classified", "Accuracy", "Total tokens", "Tokens / classified", "Retries"], c_metrics)
    status_names = sorted(set(c["zero_shot"]["status_counts"]) | set(c["few_shot"]["status_counts"]))
    c_status = table(["Status", "Zero-shot", "Few-shot"], [
        [status, c["zero_shot"]["status_counts"].get(status, 0), c["few_shot"]["status_counts"].get(status, 0)]
        for status in status_names
    ])
    confusion_path = OUT / "triage_zero_shot_confusion.csv"
    if confusion_path.exists():
        with confusion_path.open(encoding="utf-8", newline="") as file:
            reader = csv.reader(file)
            header = next(reader)
            confusion_lines = ["| True \\ Predicted | " + " | ".join(header[1:]) + " |", "| --- | " + " | ".join("---" for _ in header[1:]) + " |"]
            for row in reader:
                confusion_lines.append("| " + " | ".join(row) + " |")
        confusion_md = "\n".join(confusion_lines)
    else:
        confusion_md = "Confusion table is recorded in `outputs/triage_zero_shot_confusion.csv`."
    final_routes = c["final_route_counts"]

    d_scenarios = d["agent_scenarios"]
    def scenario_refund(row: dict) -> str:
        for step in row["steps"]:
            if step["tool_name"] in {"issue_refund", "escalate_to_human"}:
                return f"{float(step['observation']['refund_inr']):.2f}"
        return "0.00"

    scenario_table = table(
        ["Scenario", "Status", "Refund (INR)", "Model calls", "Tokens"],
        [[row["scenario_id"], row["status"], scenario_refund(row), row["llm_calls"], row["tokens_used"]]
         for row in d_scenarios],
    )
    d_demo = d["agent_demo"]
    d_eval = d["overlap_10"]
    d_eval_zero = d["overlap_0"]
    broken_question = (d_eval_zero["fact_failed_questions"] or d_eval_zero["failed_questions"] or ["(none recorded)"])[0]
    s1 = next((row for row in d_scenarios if row["scenario_id"] == "S1"), {"llm_calls": 0, "tokens_used": 0})
    s5 = next((row for row in d_scenarios if row["scenario_id"] == "S5"), {"llm_calls": 0, "tokens_used": 0})
    s6 = next((row for row in d_scenarios if row["scenario_id"] == "S6"), {"llm_calls": 0, "tokens_used": 0})
    chunk_table = table(["Policy file", "Chunks"], [[source, count] for source, count in sorted(d["chunk_counts"].items())])

    return f"""# Instamart Customer Intelligence & Support Copilot

## Overview
A single, reproducible Python repository for a fictional capstone: Part A cleans synthetic exports into SQLite, Part B segments customers and predicts lapse, Part C redacts and routes support tickets, and Part D retrieves fictional policy text and runs a bounded refund agent. All records and policies are synthetic; nothing here is real Swiggy/Instamart information. Monetary values are INR. The two supplied root scripts, `generate_data.py` and `mock_llm.py`, are included unmodified; every model call is offline through `mock_llm.py`.

## Reproduce
Python 3.11 or newer and SQLite 3.25 or newer are required. Run from the repository root. Activate the virtual environment with the command for your shell before continuing.

1. `python -m venv .venv`
2. Windows PowerShell: `.\\.venv\\Scripts\\Activate.ps1`  
   macOS/Linux: `source .venv/bin/activate`
3. `python -m pip install -r requirements.txt`
4. `python generate_data.py`
5. `python part_a_pipeline.py`
6. `python part_b_pipeline.py`
7. `python part_c_pipeline.py`
8. `python part_d_pipeline.py`
9. `python write_readme.py`

The generator's final line is `generated: customers=1200 products=40 orders_raw=13084 tickets=164`. The scripts create the required `data/clean/`, `data/instamart.db`, `outputs/`, and `state/` artifacts. The final command refreshes the numeric sections below from the generated reports.

## Part A
The generated inputs contain 1,200 customers, 40 products, 13,084 raw order rows, and 164 tickets. Cleaning standardizes customer cities, gender and age, resolves duplicate order states, removes invalid order values and writes the trusted tables to SQLite.

{part_a_table}

The detailed SQL examples are in `sql/queries.sql`; row-level clean exports are under `data/clean/`.

## Part B
**RFM snapshot (2025-10-01 00:00:00)**

{table(["Measure", "Result"], [["Customers", b_rfm["rfm_customers"]], ["Recency median / min / max", f"{b_rfm['recency_median']:.0f} / {b_rfm['recency_min']} / {b_rfm['recency_max']} days"], ["Frequency mean / max", f"{b_rfm['frequency_mean']:.2f} / {b_rfm['frequency_max']}"], ["Monetary mean / max", f"INR {b_rfm['monetary_mean']:,.2f} / INR {b_rfm['monetary_max']:,.2f}"]])}

**K-means and segment profiles**

{ k_table }

K=4 is selected: the silhouette improves sharply from K=3 to K=4, while the K=4 solution remains interpretable as four campaign groups. Inertia always falls as K increases because adding centroids can only reduce within-cluster squared distance; that monotonic decrease alone does not establish useful or stable business segments.

{profile_table}

Cluster numbers are arbitrary labels, not ranks. `Premium` is assigned to the cluster with the highest mean monetary value. PCA on the standardized RFM matrix explains {pca[0]:.3f} and {pca[1]:.3f} of variance (total {b_rfm['pca']['total_explained_variance']:.3f}).

**Lapse definition and leakage controls**

The cohort contains {b_model['cohort_size']} customers; {b_model['lapsed_count']} ({100*b_model['lapse_rate']:.2f}%) lapsed. The stratified split has {b_model['train_rows']} training rows and {b_model['test_rows']} test rows. No orders dated 2025-09-01 or later entered the features; the segment labels from task 3 were not used as features. Every encoder, scaler and resampler was fitted using training rows only, never test rows. Missing feature cells after feature construction: {b_model['missing_feature_cells']}.

The fitted scaler's training-only frequency mean is {b_model['scaler_training_mean_frequency']:.3f}, compared with the full-cohort frequency mean of {b_model['full_cohort_mean_frequency']:.3f}; they differ because the training split is a stratified sample and excludes held-out rows.

**Model selection and test evaluation**

{grid_table}

{model_table}

The all-negative baseline has lapsed-class F1 {b_model['baseline']['f1']:.3f}. The selected model is **{b_model['selected_model']}**, with lapsed-class test F1 {b_model['selected_test']['f1']:.3f}. Missing a customer who is about to lapse risks losing future orders and makes a missed intervention costly. Targeting a loyal customer unnecessarily consumes campaign budget and can train customers to wait for offers. We therefore use the selected model's lapse ranking to prioritize outreach while treating scores as decision support, not certainty.

Training-only random oversampling balanced the logistic-regression training classes to {json.dumps(b_model['balanced_train_class_counts'], sort_keys=True)}. On the unchanged test set, its lapsed precision/recall/F1 are {balanced['precision']:.3f}/{balanced['recall']:.3f}/{balanced['f1']:.3f}, versus {b_model['logistic']['test']['precision']:.3f}/{b_model['logistic']['test']['recall']:.3f}/{b_model['logistic']['test']['f1']:.3f} without oversampling. The largest movement is in recall: more true lapsers are found, at the expense of additional false-positive outreach.

**Interpretation and persisted model**

Top random-forest features:

{importance_table}

Logistic-regression coefficient directions:

{coefficient_table}

The selected pipeline is saved to `outputs/lapse_model.joblib`; reloaded predictions identical: **{b_model['joblib_predictions_identical']}**. The retention list contains {b_model['retention_list_rows']} Premium test customers, of whom {b_model['retention_list_lapsed']} actually lapsed. The Premium flag uses whole-window RFM including September, so this retention list is illustrative and is not a leakage-free evaluation.

## Part C
**Privacy and comparison**

{table(["Substitution", "Count"], [["Emails", privacy["email"]], ["Phones", privacy["phone"]], ["Order IDs", privacy["order_id"]], ["Total substitutions", privacy["total_substitutions"]], ["Residual matches after redaction", privacy["remaining_matches"]]])}

Guardrails skipped {c['zero_shot']['status_counts'].get('skipped_empty', 0)} empty tickets and flagged {c['zero_shot']['status_counts'].get('flagged_long', 0)} long tickets before any model call. Retry failures were recovered: {c['zero_shot']['retries']} zero-shot retries and {c['few_shot']['retries']} few-shot retries.

{c_table}

{c_status}

The zero-shot confusion table (rows are true labels, columns are predicted labels):

{confusion_md}

Few-shot used {c['few_shot']['tokens_per_classified_ticket']-c['zero_shot']['tokens_per_classified_ticket']:.1f} additional tokens per classified ticket and improved accuracy from {100*c['zero_shot']['accuracy_classified']:.1f}% to {100*c['few_shot']['accuracy_classified']:.1f}%. That extra prompt cost is worthwhile here because it removes {100*(c['few_shot']['accuracy_classified']-c['zero_shot']['accuracy_classified']):.1f} percentage points of classification error, especially on higher-risk ticket types. Outputs accepted as JSON or a single allowed keyword were rule-validated; empty, long, invalid, and budget-exhausted tickets were routed to human review.

The budget demonstration used {c['budget_demo']['token_budget']} tokens and exhausted the budget on {c['budget_exhausted_count']} ticket(s). Final few-shot routing counts: {json.dumps(final_routes, sort_keys=True)}. The final `outputs/triage_results.csv` has {sum(final_routes.values())} rows and columns `ticket_id, customer_id, status, category, segment_name, priority, route, tokens_used`.

## Part D
**Chunking and retrieval**

{chunk_table}

With 10-word overlap: {d_eval['doc_hits']}/{d_eval['answerable_questions']} answerable questions retrieved the expected document, {d_eval['fact_hits']}/{d_eval['answerable_questions']} answers contained the expected fact, and {d_eval['fallbacks']}/{d_eval['unanswerable_questions']} out-of-scope questions received the fallback. With zero overlap: {d_eval_zero['doc_hits']}/{d_eval_zero['answerable_questions']} expected-document hits, {d_eval_zero['fact_hits']}/{d_eval_zero['answerable_questions']} expected facts, and {d_eval_zero['fallbacks']}/{d_eval_zero['unanswerable_questions']} fallbacks. A failed zero-overlap question was: “{broken_question}”. Without overlap, related policy wording can be split across a boundary so a chunk lacks enough context for retrieval or the answer sentence is unavailable to the model. TF counts term occurrences in a chunk; IDF gives more weight to terms that occur in fewer chunks. TF-IDF is lexical, so equivalent phrasings that use different words may not match.

**Refund-agent scenarios**

{scenario_table}

The S1 token-limit demonstration ended with **{d_demo['token_limit_demo_status']}**; the zero-second timeout demonstration ended with **{d_demo['timeout_demo_status']}**. The crash-and-resume run ended with **{d_demo['crash_resume_status']}** and {d_demo['crash_resume_final_answer']}. Tools executed across both halves, in order: `{ ' -> '.join(d_demo['crash_resume_tools']) }`.

The normal ReAct run used {s1['llm_calls']} model calls for S1, {s5['llm_calls']} for S5 and {s6['llm_calls']} for S6. A plan-and-execute agent could reduce calls by proposing the whole sequence once, then replan only after an error; it would often use fewer tokens. ReAct handles unexpected observations such as `NOT_FOUND` or a cancelled order as soon as the relevant tool returns, rather than carrying out a stale plan. For refunds I prefer ReAct because each financial action should be conditioned on current database and policy evidence, and the explicit iteration/token/time limits bound the cost.

`outputs/agent_runs.jsonl` records every scenario and tool observation; `outputs/refund_ledger.jsonl` records simulated issued and escalated actions only. No network request or payment mutation is made. `state/` contains per-run checkpoints for resumption.

## Limitations and Next Steps
This is a deterministic synthetic capstone, not a production customer-support or credit model. Ticket categories are predicted by the provided offline mock, policy retrieval is lexical TF-IDF, and the order data contains deliberately planted defects. Age imputation and cluster names are analytical choices that should be reviewed before any real use. The Premium flag is a post-hoc whole-window label that includes September; therefore the retention list is illustrative and not a leakage-free evaluation. Refund issuance is simulated in a local ledger and requires human/system integration and controls before production.
"""


if __name__ == "__main__":
    (ROOT / "README.md").write_text(build_readme(), encoding="utf-8")
    print("README.md refreshed from outputs/part_*_report.json")
