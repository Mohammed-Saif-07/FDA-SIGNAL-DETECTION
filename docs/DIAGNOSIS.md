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

## Proxy Filter vs. Strict Structural Filter

The robust filter is intentionally labelled as a proxy filter. It uses
`countries_count >= 3` as a public-data approximation for source diversity
because the flattened public FAERS table used in this project does not expose a
clean independent reporter-source identifier.

The code now also emits `passes_structural_filter`, which is stricter:

- all proxy robust-filter criteria must pass
- `drug_total > case_count`
- `reaction_total > case_count`

Those last two conditions mean the 2x2 contingency table has nonzero
off-diagonal margins (`b > 0` and `c > 0`). This rejects margin-saturated pairs
where every report for a drug also contains the same reaction and every report
for a reaction also contains the same drug. These pairs can have extreme PRR/ROR
values, but they are often poor publication signals because they behave like
narrow product/event labels rather than broad pharmacovigilance signals.

The tradeoff is important: strict structural filtering is cleaner, but on the
full public FAERS table it removes many top-ranked disproportionality pairs.
That is itself a research finding: raw public-FAERS rankings are heavily shaped
by sparse and structurally narrow cells. The proxy filter preserves a rankable
watchlist while the structural flag exposes which retained candidates have
stronger 2x2-table support. This distinction follows the general caution in
Hauben and Bate (2009) that spontaneous-report data mining requires decision
support and clinical review rather than blind ranking.

## Impact

The robust filter changes the project story from:

> "PRR/ROR finds many signals."

to the more defensible research claim:

> "Naive public-FAERS PRR/ROR produces many artifact-prone signals; a transparent
> robustness layer reduces obvious artifacts while preserving reproducible
> signal-detection behavior."

This is the correct framing for a research-style applied systems paper.
