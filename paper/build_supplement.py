"""Build the paper supplement from the 81-conflict JSON dataset.

The supplement is deliberately table-heavy. It does not pretend that every
war has enough demographic data for a full contradiction-radius calculation;
instead it computes reproducible triage diagnostics for all wars and flags
which under-fudgible relation would be binding if better micro-data existed.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent / "data" / "per_war"
OUT = ROOT / "supplement.tex"


def _range(d: dict | None) -> tuple[float, float]:
    if not d:
        return 0.0, 0.0
    low = d.get("low")
    high = d.get("high")
    point = d.get("point")
    if low is None:
        low = point if point is not None else high
    if high is None:
        high = point if point is not None else low
    if low is None and high is None:
        return 0.0, 0.0
    low = float(low or 0)
    high = float(high if high is not None else low)
    if high < low:
        high = low
    return low, high


def _sum_ranges(items: list[dict]) -> tuple[float, float]:
    lo = hi = 0.0
    for item in items:
        xlo, xhi = _range(item)
        lo += xlo
        hi += xhi
    return lo, hi


@dataclass
class War:
    ident: str
    name: str
    period: str
    region: str
    mil_lo: float
    mil_hi: float
    civ_lo: float
    civ_hi: float
    ind_lo: float
    ind_hi: float
    total_lo: float
    total_hi: float
    sources: list[dict]

    @property
    def total_mid(self) -> float:
        return (self.total_lo + self.total_hi) / 2

    @property
    def civ_mid(self) -> float:
        return (self.civ_lo + self.civ_hi) / 2

    @property
    def mil_mid(self) -> float:
        return (self.mil_lo + self.mil_hi) / 2

    @property
    def civ_share_lo(self) -> float:
        return self.civ_lo / max(self.civ_lo + self.mil_hi, 1)

    @property
    def civ_share_hi(self) -> float:
        return self.civ_hi / max(self.civ_hi + self.mil_lo, 1)

    @property
    def civ_share_mid(self) -> float:
        return self.civ_mid / max(self.civ_mid + self.mil_mid, 1)

    @property
    def q_lo(self) -> float:
        return self.mil_lo / max(self.mil_lo + self.civ_hi, 1)

    @property
    def q_hi(self) -> float:
        return self.mil_hi / max(self.mil_hi + self.civ_lo, 1)

    @property
    def indirect_share(self) -> float:
        return (self.ind_lo + self.ind_hi) / max(self.civ_lo + self.civ_hi, 1)

    @property
    def total_range_ratio(self) -> float:
        return self.total_hi / max(self.total_lo, 1)

    @property
    def civ_width(self) -> float:
        return self.civ_share_hi - self.civ_share_lo


def latex_escape(s: str) -> str:
    return (
        str(s)
        .replace("\\", "")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def pct(x: float) -> str:
    if not math.isfinite(x):
        return "--"
    return f"{100*x:.0f}\\%"


def pct_range(lo: float, hi: float) -> str:
    return f"{pct(lo)}--{pct(hi)}"


def count_fmt(x: float) -> str:
    if not math.isfinite(x) or x <= 0:
        return "--"
    if x >= 1e9:
        return f"{x/1e9:.2f}B"
    if x >= 1e6:
        return f"{x/1e6:.2f}M"
    if x >= 1e3:
        return f"{x/1e3:.1f}k"
    return f"{int(round(x))}"


def diagnostic(w: War) -> str:
    if w.indirect_share >= 0.70:
        return "excess-mortality survey"
    if w.civ_share_mid >= 0.85:
        return "identified-deaths sample"
    if w.civ_share_mid <= 0.30:
        return "manpower budget"
    if w.civ_width >= 0.30:
        return "source harmonisation"
    return "demographic subsample"


def confidence_grade(w: War) -> str:
    if w.total_range_ratio >= 8 or w.civ_width >= 0.45:
        return "D"
    if w.total_range_ratio >= 4 or w.civ_width >= 0.30:
        return "C"
    if w.total_range_ratio >= 2 or w.civ_width >= 0.15:
        return "B"
    return "A"


def class_label(w: War) -> str:
    if w.indirect_share >= 0.70:
        return "indirect"
    if w.civ_share_mid >= 0.85:
        return "civilian-targeting"
    if w.civ_share_mid <= 0.30:
        return "combat-heavy"
    return "mixed"


def load_wars() -> list[War]:
    wars: list[War] = []
    for path in sorted(RESULTS.glob("*.json")):
        if path.stem.endswith("_backup") or path.stem.startswith("_"):
            continue
        raw = json.loads(path.read_text())
        if raw.get("_parse_error"):
            continue
        sides = raw.get("sides") or []
        mil_lo, mil_hi = _sum_ranges([s.get("military_killed") or {} for s in sides])
        civ_dir_lo, civ_dir_hi = _sum_ranges([s.get("civilians_killed_directly") or {} for s in sides])
        ind_items = [x for s in sides for x in (s.get("deaths_from_actions") or [])]
        ind_lo, ind_hi = _sum_ranges(ind_items)
        civ_lo, civ_hi = civ_dir_lo + ind_lo, civ_dir_hi + ind_hi
        totals = raw.get("totals") or {}
        total_lo = totals.get("grand_low")
        total_hi = totals.get("grand_high")
        if total_lo is None:
            total_lo = mil_lo + civ_lo
        if total_hi is None:
            total_hi = mil_hi + civ_hi
        total_lo = float(total_lo or 0)
        total_hi = float(total_hi or total_lo)
        if total_hi < total_lo:
            total_hi = total_lo
        regions = raw.get("regions") or raw.get("region") or ""
        if isinstance(regions, list):
            regions = ", ".join(regions[:2])
        start = raw.get("start_year") or "?"
        end = raw.get("end_year") or "2026"
        wars.append(War(
            ident=raw.get("war_id", path.stem),
            name=raw.get("name", path.stem),
            period=f"{start}--{end}",
            region=regions,
            mil_lo=mil_lo,
            mil_hi=mil_hi,
            civ_lo=civ_lo,
            civ_hi=civ_hi,
            ind_lo=ind_lo,
            ind_hi=ind_hi,
            total_lo=total_lo,
            total_hi=total_hi,
            sources=raw.get("sources") or [],
        ))
    return sorted(wars, key=lambda w: -w.total_mid)


def row(w: War, long_name: bool = False) -> str:
    name = latex_escape(w.name)
    if not long_name and len(name) > 42:
        name = name[:39] + r"\ldots{}"
    return (
        f"{name} & {w.period} & {count_fmt(w.total_mid)} "
        f"& {pct_range(w.civ_share_lo, w.civ_share_hi)} "
        f"& {pct_range(w.q_lo, w.q_hi)} "
        f"& {pct(w.indirect_share)} "
        f"& {latex_escape(diagnostic(w))} "
        f"& {confidence_grade(w)} \\\\"
    )


def write_table(lines: list[str], title: str, label: str, wars: list[War], note: str = "") -> None:
    lines += [
        rf"\subsection{{{title}}}\label{{{label}}}",
    ]
    if note:
        lines.append(note)
    lines += [
        r"\begin{scriptsize}",
        r"\begin{longtable}{p{4.7cm} c r r r r p{2.8cm} c}",
        r"\toprule",
        r"Conflict & Period & Deaths & Civ.\ share & $q$ range & Indir./Civ. & Binding relation & Grade \\",
        r"\midrule",
        r"\endhead",
    ]
    lines += [row(w) for w in wars]
    lines += [
        r"\bottomrule",
        r"\end{longtable}",
        r"\end{scriptsize}",
        "",
    ]


def main() -> None:
    wars = load_wars()
    print(f"loaded {len(wars)} wars")
    grades = {g: sum(1 for w in wars if confidence_grade(w) == g) for g in "ABCD"}
    classes = {c: sum(1 for w in wars if class_label(w) == c) for c in ["combat-heavy", "mixed", "civilian-targeting", "indirect"]}

    lines: list[str] = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[a4paper,margin=1.7cm]{geometry}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{lmodern}",
        r"\usepackage{microtype,booktabs,longtable,array,xcolor}",
        r"\usepackage{amsmath,amssymb}",
        r"\usepackage[colorlinks,linkcolor=blue!50!black,urlcolor=blue!50!black]{hyperref}",
        r"\renewcommand{\arraystretch}{1.13}",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\setlength{\emergencystretch}{3em}",
        r"\title{Supplementary Material:\\ Dataset-Wide Tables and Diagnostics}",
        r"\author{Companion to the main paper}",
        r"\date{\today}",
        r"\begin{document}",
        r"\maketitle",
        r"\begin{abstract}\raggedright\small",
        r"This supplement applies the paper's accounting framework to all 81 conflicts in the 1900--2026 dataset. "
        r"For every war we compute total death ranges, civilian share ranges, combatant share ranges $q$, the share of "
        r"civilian deaths that are indirect, a reproducible confidence grade, and the under fudgible relation most likely "
        r"to bind if a contradiction radius analysis were performed. Only conflicts with independent demographic "
        r"samples can receive a sharp contradiction radius; the rest are treated as partial identification cases.",
        r"\end{abstract}",
        r"\tableofcontents",
        "",
        r"\section{How to read the tables}",
        r"The \emph{civilian share} interval is computed conservatively as "
        r"$[D_C^{lo}/(D_C^{lo}+D_M^{hi}),\;D_C^{hi}/(D_C^{hi}+D_M^{lo})]$. "
        r"The $q$ interval is the corresponding combatant share. The \emph{binding relation} column is a triage label: "
        r"\texttt{manpower budget} means the population/combatant-stock bound is the first place to look; "
        r"\texttt{identified-deaths sample} means sex-age microdata would likely dominate; "
        r"\texttt{excess-mortality survey} means direct attribution is less informative than survey-based excess deaths; "
        r"\texttt{source harmonisation} marks conflicts where definitions, overlap, or missing high bounds dominate.",
        "",
        r"\paragraph{Confidence grade.} A: total-death range ratio $<2$ and civilian-share width $<15$ percentage points. "
        r"B: either ratio $<4$ or width $<30$pp. C: either ratio $<8$ or width $<45$pp. D: wider than that. "
        r"These grades score internal numerical precision, not moral seriousness or historical importance.",
        "",
        r"\section{Dataset summary}",
        rf"The dataset contains {len(wars)} conflicts. Confidence grades: "
        rf"A={grades['A']}, B={grades['B']}, C={grades['C']}, D={grades['D']}. "
        rf"Structural classes: combat-heavy={classes['combat-heavy']}, mixed={classes['mixed']}, "
        rf"civilian-targeting={classes['civilian-targeting']}, indirect={classes['indirect']}.",
        "",
    ]

    write_table(
        lines,
        "All 81 conflicts, ordered by midpoint total deaths",
        "tab:all",
        wars,
        "This is the master table used by the main paper's empirical overview figures.",
    )

    lines += [
        r"\section{Sources}",
        r"The table below lists the primary sources used to construct the casualty bounds for each conflict. "
        r"Full JSON files containing detailed side-by-side accounting, indirect-death breakdowns, and specific "
        r"atrocity figures are available in the companion repository.",
        r"\begin{scriptsize}",
        r"\begin{longtable}{p{3.5cm} p{12.5cm}}",
        r"\toprule",
        r"Conflict & Sources \\",
        r"\midrule",
        r"\endhead",
    ]
    
    for w in wars:
        if not w.sources:
            continue
        # Format sources compactly
        src_texts = []
        for s in w.sources:
            title = s.get("title", "").strip()
            pub = s.get("publisher", "").strip()
            if title and pub:
                src_texts.append(f"\\emph{{{latex_escape(title)}}} ({latex_escape(pub)})")
            elif title:
                src_texts.append(f"\\emph{{{latex_escape(title)}}}")
            elif pub:
                src_texts.append(latex_escape(pub))
            else:
                url = s.get("url", "")
                if url:
                    # just extract domain
                    domain = url.split("://")[-1].split("/")[0]
                    src_texts.append(latex_escape(domain))
        
        if src_texts:
            lines.append(f"{latex_escape(w.name)} & {'; '.join(src_texts)} \\\\")
            lines.append(r"\addlinespace")

    lines += [
        r"\bottomrule",
        r"\end{longtable}",
        r"\end{scriptsize}",
    ]

    lines += [
        r"\section{Reproducibility}",
        r"All tables are generated directly from the JSON files in the project results directory. "
        r"Run \texttt{python paper/build\_supplement.py} from the research folder to reproduce this PDF. "
        r"The code intentionally excludes backup directories whose names contain a period.",
        r"\end{document}",
    ]

    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
