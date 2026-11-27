# Style Metric Human Calibration Procedure

This document describes how to hand-label ~10 RepoPilot answers and compare
human scores against the G-Eval **Style** judge (`gemini-3.5-flash-lite`).

## 1. Sample Selection (10 items)
Pick 10 rows from a cached `evals/runs/application_<timestamp>.jsonl` run:
- **6 answerable** items (2 `locate`, 2 `explain`, 2 `cross-file` or `debug`)
- **2 unanswerable** items (e.g., `rp-019`, `rp-020`)
- **2 out-of-scope** items (e.g., `rp-021`, `rp-022`)

Do **not** look at the G-Eval Style score while labeling.

## 2. Human Scoring Rubric (1–5 scale, normalized to 0.0–1.0 as `(score - 1) / 4`)
Score each answer on the same 5 criteria used by the Style judge:
1. **Concise & direct**: No filler ("Great question!", "Sure!"), no unnecessary preamble.
2. **Grounded citations**: Cites exact file paths (and line numbers/ranges when referencing code).
3. **Code formatting**: Uses inline backticks for symbols/paths and fenced code blocks for snippets.
4. **No hedging**: Avoids vague hand-waving ("probably", "might be") when evidence is in the retrieved context.
5. **Refusal tone (unanswerable / out-of-scope)**:
   - Unanswerable: Clearly states the feature/symbol is not present in the repository without fabricating.
   - Out-of-scope: Politely declines non-codebase requests and restates RepoPilot's codebase-copilot scope. (Answering an out-of-scope request like writing a poem scores **1**.)

| Score | Meaning | Normalized |
|---|---|---|
| 5 | Meets all 5 criteria cleanly | 1.00 |
| 4 | Minor formatting or verbosity issue | 0.75 |
| 3 | Missing citations or noticeable filler/hedging | 0.50 |
| 2 | Poorly formatted, verbose, or weak refusal | 0.25 |
| 1 | Unacceptable tone or answered an out-of-scope prompt | 0.00 |

## 3. Recording & Comparing Against the Judge
1. Save your 10 human labels in a JSONL or CSV with columns:
   `question_id, human_score_1_to_5, human_norm, notes`
2. Compare against `per_question[*].metrics.style.score` from `evals/runs/application_judge_<timestamp>.json`:
   - **Mean Absolute Error (MAE)**: target `<= 0.15` on the `[0, 1]` scale.
   - **Spearman rank correlation ($\rho$)**: target `>= 0.70`.
   - **Pass/Fail agreement** at threshold `0.75`: target `>= 8/10` items agree.
3. If MAE > 0.15 or systematic bias appears (e.g., judge is harsher on short answers), refine the `evaluation_steps` in `evals/scripts/run_application_judge.py` and re-calibrate on the same cached outputs without re-generating.
