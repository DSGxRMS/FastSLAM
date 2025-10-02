#!/usr/bin/env python3
# plot_cones.py
# Usage: python3 plot_cones.py [path/to/cone_snapshot.csv]
# Defaults to data/cone_snapshot.csv

import sys, csv, os
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA_DIR, "cone_snapshot.csv")

# Read CSV, skipping commented meta lines
rows = []
with open(path, "r", newline="") as f:
    for line in f:
        if line.lstrip().startswith("#") or not line.strip():
            continue
        rows.append(line)
reader = csv.DictReader(rows)
pts = defaultdict(list)  # color -> [(x,y), ...]

for r in reader:
    try:
        x = float(r["x_m"]); y = float(r["y_m"]); c = r["color"].strip().lower()
    except Exception:
        continue
    pts[c].append((x, y))

# Color/style map
style = {
    "blue":          dict(marker="o",  linestyle="None", markersize=6,  label="blue",          color="blue"),
    "yellow":        dict(marker="o",  linestyle="None", markersize=6,  label="yellow",        color="gold"),
    "orange":        dict(marker="^",  linestyle="None", markersize=7,  label="orange",        color="orange"),
    "orange_large":  dict(marker="s",  linestyle="None", markersize=8,  label="orange_large",  color="orange"),
    "unknown":       dict(marker="x",  linestyle="None", markersize=6,  label="unknown",       color="black"),
}

fig = plt.figure(figsize=(7, 7))
ax = plt.gca()

# Plot each group
for c, xy in pts.items():
    xs = [p[0] for p in xy]; ys = [p[1] for p in xy]
    st = style.get(c, style["unknown"])
    ax.plot(xs, ys, **st)

ax.set_aspect("equal", adjustable="box")
ax.grid(True, linewidth=0.4, alpha=0.4)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("Cone Snapshot")
# build a clean legend (only for colors present)
handles, labels = ax.get_legend_handles_labels()
seen = set()
handles2, labels2 = [], []
for h, l in zip(handles, labels):
    if l not in seen:
        handles2.append(h); labels2.append(l); seen.add(l)
ax.legend(handles2, labels2, loc="best")

out = "cones_plot.png"
plt.tight_layout()
plt.savefig(out, dpi=150)
print(f"Saved {out}")
try:
    plt.show()
except Exception:
    pass
ṇ