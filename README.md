# Instamart Customer Intelligence & Support Copilot

> **Work in progress:** Part B–D numerical outputs have not been generated in this environment because Windows blocks NumPy's compiled extension. This repository is not yet a grading-ready submission; no missing metrics or output files are fabricated.

## Overview
A single-repository synthetic-data capstone connecting trusted data, customer segmentation and lapse prediction, support triage, and policy-grounded refund recommendations. All records and policy text are fictional. No real Swiggy/Instamart data or terms are represented; monetary values are INR. `generate_data.py` and `mock_llm.py` are the supplied root files and must remain unmodified. All LLM behavior is deterministic and offline through `mock_llm.py`.

## Reproduce
Use Python 3.11 or newer and SQLite 3.25 or newer. Run these commands in order from the repository root; activate `.venv` using the command for your shell.

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

`generate_data.py` must finish with `generated: customers=1200 products=40 orders_raw=13084 tickets=164`. The last command refreshes the numeric report sections from the generated output files.

## Part A
The raw synthetic exports are cleaned into `data/clean/customers.csv`, `data/clean/products.csv`, and `data/clean/orders.csv`. The SQLite database is `data/instamart.db`; query examples are in `sql/queries.sql`.

## Part B
RFM uses the 2025-10-01 snapshot. K-means comparison, segment profiles and PCA are reported from `outputs/`; the lapse cohort uses the specified 2025-09-01 cutoff and the label window through September 30. The classifier, training-only preprocessing, model comparison, balancing experiment, model reload check, and Premium retention list are reproduced by `part_b_pipeline.py`.

## Part C
The ticket pipeline applies the specified empty/long guards and ordered email, phone and order-ID redaction before any prompt. It compares zero-shot and few-shot classification, retries transient failures, validates model outputs, records token usage, demonstrates budget exhaustion, and routes the final few-shot run using Part B segments.

## Part D
The policy assistant uses overlapping TF-IDF chunks and the provided offline mock. The refund agent uses a tool allowlist, policy retrieval, order facts from SQLite, token/time/iteration limits, a JSONL ledger, and checkpoints in `state/`. The run also evaluates zero-overlap retrieval and demonstrates token-limit, timeout, and crash/resume behavior.

## Limitations and Next Steps
This is a deterministic synthetic learning project, not a production support or financial decision system. The model, keyword mock, lexical retriever, and policy documents are illustrative. The Premium flag is a post-hoc whole-window label that includes September, so the retention list is illustrative and not a leakage-free evaluation. Refund ledger entries are simulations only; no payment is issued or modified.
