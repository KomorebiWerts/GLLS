# Curated MMAD QA Collection

This directory contains the curated MMAD-style QA annotations used by GLLS.
It intentionally contains only `QA.json` files, not benchmark images.

Use it as `GLLS_QA_ROOT`. This should replace the raw MMAD annotation root for
evaluation; MMAD images still come from your downloaded MMAD dataset tree.

```bash
export GLLS_QA_ROOT=/path/to/GLLS/qa_collection
```

The expected layout is:

```text
qa_collection/
  DS-MVTec/<category>/QA.json
  VisA/<category>/QA.json
```

The original MMAD QA annotations contain visible annotation noise, including
wrong answers, questions with multiple valid options, questions with no valid
option, and a known DS-MVTec `pill` contamination case where QA content was
mixed with content from another dataset/category.

For reproducible GLLS runs:

- Use the downloaded MMAD `DS-MVTec/` and `VisA/` folders for images.
- Use this directory as `GLLS_QA_ROOT` for all QA annotations.
- If you keep QA files inside a separate MMAD-style tree, replace the raw MMAD
  `DS-MVTec/pill/QA.json` with `qa_collection/DS-MVTec/pill/QA.json`.

This curated collection fixes only annotation issues; it does not alter MMAD
images.
