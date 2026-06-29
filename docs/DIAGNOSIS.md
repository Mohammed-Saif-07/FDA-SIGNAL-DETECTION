# Signal Quality Diagnosis

## Finding

Naive PRR/ROR ranking on public FAERS is extremely sensitive and can surface
artifact-like drug/reaction pairs at the top of the leaderboard. Examples from
the earlier dashboard export included highly specific product names with very
large PRR/ROR values, low seriousness, and narrow reporting diversity.

This is expected behavior for unshrunk disproportionality analysis on
spontaneous-report data. It is useful as a broad signal-generation screen, but
it is not sufficient as a publication-quality ranking by itself.

## Confirmed Bug

The PySpark chi-square implementation previously multiplied large integer
margins before casting to floating point. On the current feature table this
produced negative `prr_chi_square` values for millions of rows, which is
mathematically invalid because chi-square is non-negative.

Fix:

- `spark/feature_engineering.py` now casts 2x2 table terms to `double` before
  multiplication.
- `hive/signal_detection.hql` now performs the same overflow-safe casting.

## Robust Filter

The repository now includes a conservative robust-pass filter in:

- `ml/signal_quality.py`
- `hive/signal_ranking_robust.hql`

Current robust-pass criteria:

- `case_count >= 5`
- `countries_count >= 3`
- `serious_ratio >= 0.01`
- `2 <= PRR <= 100000`
- `2 <= ROR <= 100000`
- `prr_chi_square >= 4`

`countries_count` is labelled as `source_proxy_count`. Public FAERS extracts do
not provide a clean reporter-source identifier in this flattened project table,
so this is a source-diversity proxy, not a claim of exact independent source
count.

## Impact

The robust filter changes the project story from:

> "PRR/ROR finds many signals."

to the more defensible research claim:

> "Naive public-FAERS PRR/ROR produces many artifact-prone signals; a transparent
> robustness layer reduces obvious artifacts while preserving reproducible
> signal-detection behavior."

This is the correct framing for a research-style applied systems paper.
