# Package audit — 2026-10-27

Audited the actual distributable ZIP after extraction to a separate directory.

- Full execution from saved prediction inputs: PASS.
- Five aggregate CSVs: all numeric columns agree with accepted references within 1e-10.
- Six generated LaTeX files (four tables, combined tables, preview): byte-identical to the packaged examples.
- All 37 input file records retain SHA-256 provenance; no raw patient data are bundled.
- No local absolute paths or obsolete 2,000/10,000 bootstrap settings found.
- Dependencies and README commands match the extracted directory structure.
- Scope is the four added analyses; no threshold-selection audit, training code, or full manuscript is included.
- The verifier now requires the complete ordered column schema, bootstrap=1000, and seed=20261024.
- Negative tests: missing CI column, altered CI value, and wrong bootstrap count all correctly fail verification.

No statistical computations or accepted results changed in this audit. This package reads saved prediction files; it neither retrains models nor updates manuscript files. Server deployment has not occurred.
