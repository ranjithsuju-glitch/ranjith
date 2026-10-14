# Instamart Customer Intelligence & Support Copilot

## Overview
A single, reproducible Python repository for a fictional capstone: Part A cleans synthetic exports into SQLite, Part B segments customers and predicts lapse, Part C redacts and routes support tickets, and Part D retrieves fictional policy text and runs a bounded refund agent. All records and policies are synthetic; nothing here is real Swiggy/Instamart information. Monetary values are INR. The two supplied root scripts, `generate_data.py` and `mock_llm.py`, are included unmodified; every model call is offline through `mock_llm.py`.

## Reproduce
Python 3.11 or newer and SQLite 3.25 or newer are required. Run from the repository root. Activate the virtual environment with the command for your shell before continuing.

1. `python -m venv .venv`
2. Windows PowerShell: `.\.venv\Scripts\Activate.ps1`  
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

| Measure | Result |
| --- | --- |
| Raw Rows | 13084 |
| Duplicate Rows Removed | 775 |
| Invalid Rows Removed | 0 |
| Order Amounts Imputed | 112 |
| Completed Delivery Values Set Missing | 774 |
| Clean Rows | 12309 |
| Completed Rows | 11110 |
| Cancelled Rows | 1199 |
| Clean Customers | 1200 |
| Clean Products | 40 |

The detailed SQL examples are in `sql/queries.sql`; row-level clean exports are under `data/clean/`.

## Part B
**RFM snapshot (2025-10-01 00:00:00)**

| Measure | Result |
| --- | --- |
| Customers | 1175 |
| Recency median / min / max | 19 / 0 / 180 days |
| Frequency mean / max | 9.46 / 33 |
| Monetary mean / max | INR 2,552.06 / INR 14,766.55 |

**K-means and segment profiles**

| K | Inertia | Silhouette |
| --- | --- | --- |
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

K=4 is selected: the silhouette improves sharply from K=3 to K=4, while the K=4 solution remains interpretable as four campaign groups. Inertia always falls as K increases because adding centroids can only reduce within-cluster squared distance; that monotonic decrease alone does not establish useful or stable business segments.

| Segment ID | Name | Size | Mean recency | Mean frequency | Mean monetary (INR) | Recommended action | Premium |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | Occasional | 436 | 28.86 | 3.77 | 1057.62 | Encourage a second monthly order with a focused category offer. | False |
| 1 | Premium | 138 | 24.12 | 6.77 | 7129.70 | Protect loyalty with an exclusive retention touchpoint. | True |
| 2 | Frequent Essentials | 419 | 8.77 | 19.40 | 3410.02 | Use replenishment reminders and relevant basket bundles. | False |
| 3 | At Risk | 182 | 109.04 | 2.20 | 685.99 | Test a timely win-back offer and monitor the next order. | False |

Cluster numbers are arbitrary labels, not ranks. `Premium` is assigned to the cluster with the highest mean monetary value. PCA on the standardized RFM matrix explains 0.634 and 0.218 of variance (total 0.852).

**Lapse definition and leakage controls**

The cohort contains 1156 customers; 443 (38.32%) lapsed. The stratified split has 924 training rows and 232 test rows. No orders dated 2025-09-01 or later entered the features; the segment labels from task 3 were not used as features. Every encoder, scaler and resampler was fitted using training rows only, never test rows. Missing feature cells after feature construction: 0.

The fitted scaler's training-only frequency mean is 8.070, compared with the full-cohort frequency mean of 8.029; they differ because the training split is a stratified sample and excludes held-out rows.

**Model selection and test evaluation**

| Model | Best parameters | Best CV F1 |
| --- | --- | --- |
| Logistic regression | {"model__C": 100} | 0.649 |
| Random forest | {"model__max_depth": 5, "model__min_samples_leaf": 10} | 0.686 |

| Model | Confusion matrix [[TN, FP], [FN, TP]] | Lapsed precision | Lapsed recall | Lapsed F1 | Accuracy |
| --- | --- | --- | --- | --- | --- |
| Always not lapsed | [[143, 0], [89, 0]] | 0.000 | 0.000 | 0.000 | 0.616 |
| Logistic regression | [[104, 39], [31, 58]] | 0.598 | 0.652 | 0.624 | - |
| Random forest | [[111, 32], [32, 57]] | 0.640 | 0.640 | 0.640 | - |

The all-negative baseline has lapsed-class F1 0.000. The selected model is **random_forest**, with lapsed-class test F1 0.640. Missing a customer who is about to lapse risks losing future orders and makes a missed intervention costly. Targeting a loyal customer unnecessarily consumes campaign budget and can train customers to wait for offers. We therefore use the selected model's lapse ranking to prioritize outreach while treating scores as decision support, not certainty.

Training-only random oversampling balanced the logistic-regression training classes to {"0": 570, "1": 570}. On the unchanged test set, its lapsed precision/recall/F1 are 0.557/0.764/0.645, versus 0.598/0.652/0.624 without oversampling. The largest movement is in recall: more true lapsers are found, at the expense of additional false-positive outreach.

**Interpretation and persisted model**

Top random-forest features:

| Feature | Importance |
| --- | --- |
| numeric__frequency | 0.309 |
| numeric__monetary | 0.269 |
| numeric__recency | 0.125 |
| numeric__avg_delivery_minutes | 0.061 |
| numeric__avg_discount_pct | 0.053 |

Logistic-regression coefficient directions:

| Feature | Coefficient | Direction | Business reading |
| --- | --- | --- | --- |
| recency | -0.000 | negative | Longer time since the last order is associated with lower lapse odds in this fitted model. |
| frequency | -1.862 | negative | More completed orders are associated with lower lapse odds. |
| monetary | 0.367 | positive | Higher historical spend is associated with higher lapse odds in this fitted model. |

The selected pipeline is saved to `outputs/lapse_model.joblib`; reloaded predictions identical: **True**. The retention list contains 10 Premium test customers, of whom 2 actually lapsed. The Premium flag uses whole-window RFM including September, so this retention list is illustrative and is not a leakage-free evaluation.

## Part C
**Privacy and comparison**

| Substitution | Count |
| --- | --- |
| Emails | 55 |
| Phones | 91 |
| Order IDs | 132 |
| Total substitutions | 278 |
| Residual matches after redaction | 0 |

Guardrails skipped 4 empty tickets and flagged 4 long tickets before any model call. Retry failures were recovered: 19 zero-shot retries and 19 few-shot retries.

| Run | Classified | Accuracy | Total tokens | Tokens / classified | Retries |
| --- | --- | --- | --- | --- | --- |
| Zero Shot | 147 | 85.7% | 9400 | 63.9 | 19 |
| Few Shot | 147 | 100.0% | 12832 | 87.3 | 19 |

| Status | Zero-shot | Few-shot |
| --- | --- | --- |
| classified_json | 135 | 135 |
| classified_keyword | 12 | 12 |
| flagged_invalid_output | 9 | 9 |
| flagged_long | 4 | 4 |
| skipped_empty | 4 | 4 |

The zero-shot confusion table (rows are true labels, columns are predicted labels):

| True \ Predicted | billing | delivery | technical | fraud |
| --- | --- | --- | --- | --- |
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| technical | 10 | 0 | 29 | 0 |
| fraud | 11 | 0 | 0 | 23 |

Few-shot used 23.3 additional tokens per classified ticket and improved accuracy from 85.7% to 100.0%. That extra prompt cost is worthwhile here because it removes 14.3 percentage points of classification error, especially on higher-risk ticket types. Outputs accepted as JSON or a single allowed keyword were rule-validated; empty, long, invalid, and budget-exhausted tickets were routed to human review.

The budget demonstration used 6000 tokens and exhausted the budget on 56 ticket(s). Final few-shot routing counts: {"human_review": 17, "premium_desk": 11, "security_team": 34, "support_queue": 102}. The final `outputs/triage_results.csv` has 164 rows and columns `ticket_id, customer_id, status, category, segment_name, priority, route, tokens_used`.

## Part D
**Chunking and retrieval**

| Policy file | Chunks |
| --- | --- |
| delivery_policy.txt | 3 |
| escalation_policy.txt | 3 |
| payment_policy.txt | 3 |
| privacy_policy.txt | 2 |
| refund_policy.txt | 4 |

With 10-word overlap: 12/12 answerable questions retrieved the expected document, 12/12 answers contained the expected fact, and 2/2 out-of-scope questions received the fallback. With zero overlap: 12/12 expected-document hits, 10/12 expected facts, and 2/2 fallbacks. A failed zero-overlap question was: “What delivery time makes a completed order eligible for a full refund?”. Without overlap, related policy wording can be split across a boundary so a chunk lacks enough context for retrieval or the answer sentence is unavailable to the model. TF counts term occurrences in a chunk; IDF gives more weight to terms that occur in fewer chunks. TF-IDF is lexical, so equivalent phrasings that use different words may not match.

**Refund-agent scenarios**

| Scenario | Status | Refund (INR) | Model calls | Tokens |
| --- | --- | --- | --- | --- |
| S1 | REFUND | 128.25 | 5 | 869 |
| S2 | REFUND | 135.00 | 5 | 869 |
| S3 | ESCALATE | 1999.00 | 5 | 856 |
| S4 | DENIED | 0.00 | 4 | 595 |
| S5 | DENIED | 0.00 | 3 | 344 |
| S6 | NOT_FOUND | 0.00 | 2 | 102 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 2751 |

The S1 token-limit demonstration ended with **FALLBACK_TOKEN_LIMIT**; the zero-second timeout demonstration ended with **FALLBACK_TIMEOUT**. The crash-and-resume run ended with **REFUND** and REFUND: INR 128.25 issued. Tools executed across both halves, in order: `get_order -> search_policy -> calculate_refund -> issue_refund`.

The normal ReAct run used 5 model calls for S1, 3 for S5 and 2 for S6. A plan-and-execute agent could reduce calls by proposing the whole sequence once, then replan only after an error; it would often use fewer tokens. ReAct handles unexpected observations such as `NOT_FOUND` or a cancelled order as soon as the relevant tool returns, rather than carrying out a stale plan. For refunds I prefer ReAct because each financial action should be conditioned on current database and policy evidence, and the explicit iteration/token/time limits bound the cost.

`outputs/agent_runs.jsonl` records every scenario and tool observation; `outputs/refund_ledger.jsonl` records simulated issued and escalated actions only. No network request or payment mutation is made. `state/` contains per-run checkpoints for resumption.

## Limitations and Next Steps
This is a deterministic synthetic capstone, not a production customer-support or credit model. Ticket categories are predicted by the provided offline mock, policy retrieval is lexical TF-IDF, and the order data contains deliberately planted defects. Age imputation and cluster names are analytical choices that should be reviewed before any real use. The Premium flag is a post-hoc whole-window label that includes September; therefore the retention list is illustrative and not a leakage-free evaluation. Refund issuance is simulated in a local ledger and requires human/system integration and controls before production.
