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

Mind the sign convention, which differs between the two:

* the paper's `Δc` is `d_baseline - d_intervened`, so **positive is better**;
* `paper_table.csv`'s `mean_error_delta_vs_baseline_km` is the error *increase*,
  `d_intervened - d_baseline`, so **negative is better**.

Both are correct for what their names say; they are not interchangeable.

The current Table 1 comes from the per-layer discovery/holdout search, not from
these scripts — see `../experiments/README.md`. The scripts are kept because
they document the earlier full-dataset runs that are committed here.

## Live UI figure

`figures/ui-demo-live.jpg` is an unaltered full-page capture of the running UI.
`ui-demo-controls.jpg` and `ui-demo-results.jpg` are browser-captured crops of
that same session, annotated with numbered callouts in LaTeX.

The input is `392426226_fde8ad65f3_166_32377115@N00.jpg` from the local
`Thesis/Data/demo` directory. Ground truth in its accompanying `gt.txt` is
`53.306339, -6.441485`. A manually drawn uniform region activates 84 patches.
Its normalised rectangle is approximately `x=0.0938825, y=0.138375,
w=0.28195, h=0.819024` in the original image.
Layer 2 uses `a=-2, b=0`; all other layers have zero biases; top-k is 1.
The live baseline is `44.1089, -77.5489` (Quinte West) and the intervened
prediction is `45.4593, -75.6437` (Ottawa). The UI reports 212.5 km great-circle
displacement. This is a demonstration example, separate from Table 1's selected
configuration and held-out evaluation.
