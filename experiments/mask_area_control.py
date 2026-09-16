"""Does the intervention's effect just track how much of the image was masked?

    python experiments/mask_area_control.py [RUN_DIR]

Reads a run's per_image.csv and relates the fraction of patches inside the
region to the geodesic distance between the baseline and intervened
predictions. Reports Spearman rho and a coverage-quartile breakdown.

The naive objection to a region-conditioned intervention is that a bigger mask
perturbs more of the image and therefore moves the prediction further. It does
not: a mask covering almost every patch adds nearly the same logit bias to
every patch key, and softmax is invariant to a constant shift, so the
intervention cancels. The relationship is negative, not positive.

Stdlib only -- the ranking and the correlation are a few lines each.
"""
import csv
import math
import statistics
import sys
from pathlib import Path

DEFAULT_RUN = (Path(__file__).resolve().parent.parent / 'results' /
               'img2gps3k_wedetect_union_all_layers_a2_bneg2_threshold_0.4_g4_hf')
EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(h))


def ranks(values):
    """Average ranks, so ties -- of which displacement has many at 0 km -- don't
    bias the correlation towards whichever order the file happened to use."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            out[order[k]] = shared
        i = j + 1
    return out


def pearson(xs, ys):
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    den = math.sqrt(sum(d * d for d in dx) * sum(d * d for d in dy))
    return sum(a * b for a, b in zip(dx, dy)) / den if den else 0.0


def spearman(xs, ys):
    return pearson(ranks(xs), ranks(ys))


def load(run_dir):
    """Active-mask rows only: an empty mask is an exact no-op and carries no
    information about area."""
    rows = []
    with open(Path(run_dir) / 'per_image.csv', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            if not row.get('union_lat') or not row.get('union_patch_count'):
                continue
            patches = int(float(row['union_patch_count']))
            if patches <= 0:
                continue
            rows.append((
                patches,
                float(row['union_coverage']),
                haversine_km(float(row['baseline_lat']), float(row['baseline_lon']),
                             float(row['union_lat']), float(row['union_lon'])),
            ))
    return rows


def report(run_dir=DEFAULT_RUN):
    rows = load(run_dir)
    coverage = [r[1] for r in rows]
    displacement = [r[2] for r in rows]
    rho = spearman(coverage, displacement)

    print(f'run                 {Path(run_dir).name}')
    print(f'active-mask images  {len(rows)}')
    print(f'mean patches        {statistics.fmean(r[0] for r in rows):.1f} of 256')
    print(f'mean coverage       {statistics.fmean(coverage):.3f}')
    print(f'mean displacement   {statistics.fmean(displacement):.1f} km')
    print(f'coverage vs displacement, Spearman rho = {rho:+.3f}')
    print()
    print('quartile  coverage      n   mean disp   median disp   zero shift')
    cuts = statistics.quantiles(coverage, n=4)
    edges = [0.0] + cuts + [1.0]
    for q in range(4):
        low, high = edges[q], edges[q + 1]
        group = [r for r in rows if (low < r[1] <= high) or (q == 0 and r[1] <= high)]
        zero = sum(1 for r in group if r[2] < 1e-6) / len(group)
        print(f'Q{q + 1}        {low:.2f}-{high:.2f}  {len(group):5d}  '
              f'{statistics.fmean(r[2] for r in group):8.1f} km  '
              f'{statistics.median(r[2] for r in group):9.1f} km  '
              f'{100 * zero:9.1f}%')
    return rho


def _self_check():
    assert ranks([3, 1, 2]) == [3.0, 1.0, 2.0]
    assert ranks([1, 1, 2]) == [1.5, 1.5, 3.0]
    assert abs(spearman([1, 2, 3, 4], [1, 2, 3, 4]) - 1.0) < 1e-12
    assert abs(spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-12
    assert abs(haversine_km(0, 0, 0, 1) - 111.19) < 0.01
    assert abs(haversine_km(10, 20, 10, 20)) < 1e-9


if __name__ == '__main__':
    _self_check()
    report(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RUN)
