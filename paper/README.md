# Manuscript and demonstration package

Independent diagnosis
validation was not completed and is not claimed. Existing outcome/judge reporting limits are retained.

## Build

From the repository root:

```sh
cd paper
tectonic -o ../output/pdf main.tex
```

Expected: exit 0 and `output/pdf/main.pdf`, with four content pages and references on page five. Tectonic may download standard TeX packages on first use.
The two table bodies in `generated/` were generated from `results/metrics.json` and
`results/repair_study_v1/summary.json`. The source hashes are in `generated/source_hashes.json`.
No experiments need rerunning to typeset this draft.

`llncs.cls` and `splncs04.bst` are unmodified official Springer LNCS v2.26, downloaded 3 Nov 2026 from
https://mirrors.ctan.org/macros/latex/contrib/llncs.zip. Attribution/license: `LNCS_LICENSE.md` (CC BY 4.0).
Fig. 1 is `figures/ui_whatif.png`, a screenshot of the running workbench (Tool health → what-if editor).
