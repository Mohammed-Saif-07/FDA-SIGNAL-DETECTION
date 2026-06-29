# FDA Warning Reference Data Dictionary

`data/reference/fda_warnings.csv` is the hand-curated ground-truth file used for
backtesting. It is intentionally small and auditable; entries should not be
added unless a public FDA or regulator source can support the drug, reaction,
and warning date.

| Column | Meaning |
| --- | --- |
| `drug_name` | Normalized uppercase drug/product name used for joins. |
| `reaction_term` | Normalized MedDRA-style adverse reaction term used for joins. |
| `warning_date` | Date of FDA warning, safety communication, boxed-warning action, or recall. |
| `warning_type` | Regulatory action type, for example `RECALL`, `BOXED_WARNING`, or `SAFETY_COMMUNICATION`. |
| `recall_date` | Recall/withdrawal date when applicable. Empty for non-recall safety actions. |
| `source_url` | Public source URL supporting the entry. Prefer FDA pages. |
| `notes` | Short curation note explaining the mapping. |

## Curation Rules

- Do not fabricate entries.
- Prefer official FDA pages; if not available, use a stable public regulator or
  manufacturer safety source and document it in `notes`.
- Use exact normalized strings that can reasonably match FAERS drug and reaction
  terms.
- Keep multiple rows for one drug if the FDA action covers multiple clinically
  distinct reactions.
- Treat this file as a benchmark, not as a complete registry of all FDA actions.

## Current Limitation

The file currently contains 56 rows, but only 7 entries are after the
`2020-12-31` validation cutoff. This is enough to demonstrate the backtesting
workflow, but not enough to support a strong general performance claim.
