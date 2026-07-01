# Submission Plan

## Recommended First Target: ML4H Workshop

Best fit for the current contribution: an open-source, reproducible FAERS
workflow with classical pharmacovigilance baselines, Docker/conda replay, and
honest small-sample evaluation. The strongest defensible result is BCPNN IC025
catching 2 of 7 future FDA warning pairs at the 2020 cutoff, with wide bootstrap
confidence intervals.

## arXiv Fallback

Upload the compiled paper and source bundle as a preprint after final manual
review. Use arXiv only for visibility; do not describe the work as clinically
validated or causal.

## Candidate: JAMIA Open

Possible if the warning reference set is expanded and FDA source URLs are
deep-link verified. A journal submission also needs stronger terminology
normalization and a more complete discussion of spontaneous-report bias.

## Stretch: Drug Safety

Best domain fit, but the current validation sample is small and the EBGM
baseline is simplified rather than production MGPS. Treat as a later target
after MedDRA grouping, FDA warning-source verification, and a larger temporal
validation set.

## Before Submission Checklist

- Replace paper author email and ORCID placeholders.
- Spot-check every BibTeX entry in Google Scholar or PubMed.
- Verify at least the caught warning and top missed warnings with FDA deep-link
  URLs.
- Compile on Overleaf and inspect all figures/tables for overflow.
- Run `make reproduce-paper`, `make compare-methods`, and `make test`.
- Archive a GitHub release on Zenodo and update the DOI badge/CITATION file.
