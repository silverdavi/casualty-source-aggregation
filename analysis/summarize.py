"""Aggregate the per-war JSON files into:
  - data/summary.json   machine-readable index
  - data/SUMMARY.md     human-readable rolled-up table
  - data/totals.csv     flat CSV: war,side,bucket,low,high,notes
"""
from __future__ import annotations
import csv, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PER_WAR = DATA / "per_war"

def fmt(n):
    if n is None: return "—"
    try: return f"{int(n):,}"
    except: return str(n)

def midpoint(lo, hi):
    if lo is None and hi is None: return None
    if lo is None: return hi
    if hi is None: return lo
    return (lo + hi) / 2

wars_meta = json.loads((DATA / "wars_input.json").read_text())["wars"]
order = {w["id"]: i for i, w in enumerate(wars_meta)}

rows = []
index = []
csv_rows = []

for d in sorted(PER_WAR.glob("*.json"), key=lambda p: order.get(p.stem, 999)):
    j = json.loads(d.read_text())
    wid = j.get("war_id", d.stem)
    name = j.get("name", wid)
    totals = j.get("totals", {}) or {}
    g_lo, g_hi = totals.get("grand_low"), totals.get("grand_high")
    m_lo, m_hi = totals.get("military_low"), totals.get("military_high")
    c_lo, c_hi = totals.get("civilian_low"), totals.get("civilian_high")

    sides = j.get("sides") or []
    n_atroc = len(j.get("key_atrocities") or [])
    n_sources = len(j.get("sources") or [])

    index.append({
        "war_id": wid, "name": name,
        "start_year": j.get("start_year"), "end_year": j.get("end_year"),
        "ongoing": j.get("ongoing"),
        "total_low": g_lo, "total_high": g_hi,
        "military_low": m_lo, "military_high": m_hi,
        "civilian_low": c_lo, "civilian_high": c_hi,
        "n_sides": len(sides), "n_key_atrocities": n_atroc,
        "n_sources": n_sources,
        "midpoint": midpoint(g_lo, g_hi),
    })
    rows.append((wid, name, j, totals))

    for s in sides:
        sname = s.get("name", "?")
        for bucket in ("military_killed", "civilians_killed_directly"):
            b = s.get(bucket) or {}
            csv_rows.append([wid, name, sname, bucket,
                             b.get("low"), b.get("high"),
                             (b.get("notes") or "").replace("\n"," ")[:300]])
        for ind in (s.get("deaths_from_actions") or []):
            csv_rows.append([wid, name, sname,
                             f"deaths_from_actions:{ind.get('type','?')}",
                             ind.get("low"), ind.get("high"),
                             (ind.get("notes") or "").replace("\n"," ")[:300]])

# CSV
with (DATA / "totals.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["war_id","war_name","side","bucket","low","high","notes"])
    w.writerows(csv_rows)

# JSON index
(DATA / "summary.json").write_text(json.dumps({
    "total_wars": len(index),
    "estimated_grand_total_low":  sum((x["total_low"] or 0) for x in index),
    "estimated_grand_total_high": sum((x["total_high"] or 0) for x in index),
    "wars": index,
}, indent=2))

# Markdown
md = ["# Wars 1900-2026 — Casualty Summary\n"]
md.append(f"_{len(index)} conflicts; per-side military / civilian-direct / "
         "indirect death buckets, low--high ranges with sources._\n")
md.append("Numbers are *low--high* ranges from the most credible sources "
         "available. See `per_war/<war_id>.json` for sources, per-side "
         "breakdowns, and indirect (camp/famine/blockade/disease) deaths.\n")

md.append("## Aggregate (sum of mid-point estimates)\n")
mid_total = sum((x["midpoint"] or 0) for x in index)
lo_total  = sum((x["total_low"]  or 0) for x in index)
hi_total  = sum((x["total_high"] or 0) for x in index)
md.append(f"- Sum of low estimates: **{int(lo_total):,}**")
md.append(f"- Sum of high estimates: **{int(hi_total):,}**")
md.append(f"- Sum of midpoints: **{int(mid_total):,}**\n")
md.append("(These sums double-count overlapping conflicts — e.g. WWII theaters "
         "vs the Holocaust, Iraq War vs anti-ISIS war — and should not be read "
         "as 'total deaths from all 20th–21st century war'.)\n")

md.append("## By total death toll (high estimate)\n")
md.append("| War | Years | Total (low–high) | Mil (low–high) | Civ (low–high) |")
md.append("|---|---|---:|---:|---:|")
for x in sorted(index, key=lambda y: -(y["total_high"] or 0)):
    yrs = f"{x['start_year']}–{x['end_year'] or 'now'}"
    md.append(f"| [{x['name']}](per_war/{x['war_id']}.json) | {yrs} | "
              f"{fmt(x['total_low'])}–{fmt(x['total_high'])} | "
              f"{fmt(x['military_low'])}–{fmt(x['military_high'])} | "
              f"{fmt(x['civilian_low'])}–{fmt(x['civilian_high'])} |")

md.append("\n## Chronological\n")
md.append("| War | Years | Total (low–high) |")
md.append("|---|---|---:|")
for x in sorted(index, key=lambda y: (y["start_year"] or 0)):
    yrs = f"{x['start_year']}–{x['end_year'] or 'now'}"
    md.append(f"| [{x['name']}](per_war/{x['war_id']}.json) | {yrs} | "
              f"{fmt(x['total_low'])}–{fmt(x['total_high'])} |")

(DATA / "SUMMARY.md").write_text("\n".join(md))
print(f"Wrote {DATA}/SUMMARY.md, summary.json, totals.csv ({len(index)} wars)")
print(f"Sum of high estimates: {int(hi_total):,}")
