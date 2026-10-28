"""Per-coverage breakdown of results/region_specificity (bins fixed before the run).

    python experiments/analyze_region_specificity.py [results/region_specificity]
"""
import json, sys, numpy as np, pandas as pd
root = sys.argv[1] if len(sys.argv) > 1 else "results/region_specificity"
def boot(v, n=10000):
    rng = np.random.default_rng(42); m = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return np.percentile(m, [2.5, 97.5])
M = ["one_minus_cosine", "changed", "displacement_km", "delta_km"]
for mode, pairs in (("box", [("target", "background")]), ("union", [("target", "shift"), ("target", "scatter"), ("shift", "scatter")])):
    s = json.load(open(f"{root}/{mode}/summary.json"))
    print(f"\n######## {mode}: imgs {s['evaluated_images']} skipped {s['units_skipped_no_background']} cfg {s['config']['layers']} a={s['config']['a']:.3f} b={s['config']['b']:.3f}")
    d = pd.read_csv(f"{root}/{mode}/draws.csv")
    d["delta_km"] = d.baseline_distance_km - d.distance_km
    d["one_minus_cosine"] = 1 - d.cosine; d["changed"] = d.changed.astype(float)
    d["cov"] = d.unit_patches / 256
    d["bin"] = pd.cut(d["cov"], [0, .1, .25, .5, .8, 1.0])
    for sub_name, sub in (("all", d), ("holdout", d[d.split == "holdout"])):
        print(f"--- {sub_name}: images {sub.name.nunique()} units {sub.groupby(['name','unit']).ngroups}")
        print(sub.groupby("arm")[M + ["distance_km"]].mean().round(4).to_string())
        pu = sub.groupby(["name", "unit", "arm"])[M].mean().unstack("arm")
        binof = sub.groupby(["name", "unit"])["bin"].first()
        for l, r in pairs:
            for label, mask in [("ALL", slice(None))] + ([(str(b), binof == b) for b in binof.cat.categories] if mode == "union" else []):
                part = pu[mask] if label != "ALL" else pu
                bm = binof[mask] if label != "ALL" else binof
                out = []
                for m in M:
                    v = (part[(m, l)] - part[(m, r)]).groupby(level="name").mean().to_numpy()
                    lo, hi = boot(v); sig = "*" if lo > 0 or hi < 0 else " "
                    out.append(f"{m[:8]} {v.mean():+.4g} [{lo:+.3g},{hi:+.3g}]{sig}")
                print(f"  {l}-{r} {label:12s} n={len(part):4d} | " + " | ".join(out))
    if mode == "box":
        sub = d
        pu = sub.groupby(["name","unit","arm"])[M].mean().unstack("arm")
        sz = sub.groupby(["name","unit"]).unit_patches.first()
        for lab, msk in (("<=4", sz<=4), ("5-12", (sz>4)&(sz<=12)), (">12", sz>12)):
            p = pu[msk]; out=[]
            for m in M:
                v = (p[(m,"target")]-p[(m,"background")]).groupby(level="name").mean().to_numpy(); lo,hi=boot(v)
                out.append(f"{m[:8]} {v.mean():+.4g} [{lo:+.3g},{hi:+.3g}]{'*' if lo>0 or hi<0 else ' '}")
            print(f"  size {lab:5s} units={len(p):4d} | " + " | ".join(out))
