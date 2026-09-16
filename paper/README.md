# Paper source

`latexmk -pdf main.tex` builds `main.pdf`. LNCS class files (`llncs.cls`,
`splncs04.bst`) are vendored because Springer does not host them on CTAN in a
stable location.

* `sections/` — one file per section, `\input` by `main.tex` in order
* `figures/` — `video-cards.html` renders the three title cards of the
  demonstration video; the PNGs are its exported frames
* `tables/paper_table.csv`, `tables/paper_facts.json` — aggregate metrics of
  the WeDetect proposal runs in `../results/`
* `scripts/collect_results.py` → writes `tables/paper_table.csv` from
  `../results/*/per_image.csv`; `scripts/make_tables.py` renders it to LaTeX

The current Table 1 comes from the per-layer discovery/holdout search, not from
these scripts — see `../experiments/README.md`. The scripts are kept because
they document the earlier full-dataset runs that are committed here.
