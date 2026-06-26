# Curated MMAD QA Collection

This directory contains the MMAD-style QA files used by GLLS. It contains only
`QA.json` files, not benchmark images.

Use it as `GLLS_QA_ROOT`; keep the images in your downloaded MMAD tree.

```bash
export GLLS_QA_ROOT=/path/to/GLLS/qa_collection
```

The expected layout is:

```text
qa_collection/
  DS-MVTec/<category>/QA.json
  VisA/<category>/QA.json
```

The raw MMAD QA files contain known label and option errors.

For reproducible GLLS runs:

- Use the downloaded MMAD `DS-MVTec/` and `VisA/` folders for images.
- Use this directory as `GLLS_QA_ROOT` for all QA annotations.
- Before evaluating DS-MVTec `pill`, fix the image folder if needed: affected
  MMAD copies have `metal_nut` images in `DS-MVTec/pill/image/good/`. Replace
  that folder with official MVTec-AD `pill/test/good/` images.

These QA files do not alter MMAD images.
