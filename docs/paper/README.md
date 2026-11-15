# Research manuscript

`paper.tex` is the editable manuscript in the IEEEtran conference format. `paper.pdf` is its reviewed rendering. The author block uses the project-team name because the complete author list and order have not been supplied. Replace that block with the actual authors, affiliations and institutional identifiers before submission. Do not infer the paper's author list from Git commit authorship.

## Rebuild

Use a LaTeX distribution with the IEEEtran class (TeX Live's `texlive-publishers`, or an IEEE conference project in Overleaf). Upload `paper.tex`, `grouped_table.tex` and `results_plot.pdf` together. In this folder run:

```sh
pdflatex paper.tex
pdflatex paper.tex
```

The two result tables use the standalone rerun and the separately tuned road-link holdout. The paper distinguishes these protocols and does not claim prospective validation, calibrated risks or deployment safety.
