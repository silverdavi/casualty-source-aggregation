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
    mil_incomplete: bool = False
    civ_incomplete: bool = False

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
        # Indirect deaths are a subset of civilian deaths (civ = direct + indirect),
        # so this share cannot exceed 1. When a curated civilian total in the
        # totals block undercounts the summed indirect component, the raw ratio
        # can spuriously exceed 100%; clamp it to keep the displayed share sane.
        ratio = (self.ind_lo + self.ind_hi) / max(self.civ_lo + self.civ_hi, 1)
        return min(1.0, ratio)

    @property
    def total_range_ratio(self) -> float:
        return self.total_hi / max(self.total_lo, 1)

    @property
    def id_share_lo(self) -> float:
        """Lower end of the identified civilian-share interval: an
        incompletely attributed military class only bounds the share above."""
        return 0.0 if self.mil_incomplete else self.civ_share_lo

    @property
    def id_share_hi(self) -> float:
        return 1.0 if self.civ_incomplete else self.civ_share_hi

    @property
    def civ_width(self) -> float:
        return self.id_share_hi - self.id_share_lo


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


def effective_grade(w: War) -> str:
    if w.mil_incomplete and w.civ_incomplete:
        return "n.a."
    return confidence_grade(w)


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

        def _has_num(d: dict | None) -> bool:
            return bool(d) and any(d.get(k) is not None for k in ("low", "high", "point"))

        def _open_above(d: dict | None) -> bool:
            if not _has_num(d):
                return True
            return d.get("high") is None and d.get("point") is None

        # A class is incompletely attributed when its ceiling is unstated for
        # some side; a curated *_high in the totals block cures this, while a
        # floor-only totals entry does not.
        mil_open = any(_open_above(s.get("military_killed")) for s in sides)
        civ_open = any(
            _open_above(s.get("civilians_killed_directly"))
            and not any(_has_num(a) and not _open_above(a) for a in (s.get("deaths_from_actions") or []))
            for s in sides
        )
        totals = raw.get("totals") or {}
        # Prefer the curated military/civilian split in totals when present:
        # side-level fields are often null for wars where sources do not
        # attribute deaths per side, which would otherwise zero out one class.
        t_ml, t_mh = totals.get("military_low"), totals.get("military_high")
        if t_ml is not None or t_mh is not None:
            mil_lo = float(t_ml if t_ml is not None else t_mh)
            # A floor-only totals entry keeps the side-derived ceiling if larger.
            mil_hi = float(t_mh) if t_mh is not None else max(mil_hi, mil_lo)
        t_cl, t_ch = totals.get("civilian_low"), totals.get("civilian_high")
        if t_cl is not None or t_ch is not None:
            civ_lo = float(t_cl if t_cl is not None else t_ch)
            civ_hi = float(t_ch) if t_ch is not None else max(civ_hi, civ_lo)
        mil_no_ceiling = t_mh is None and (mil_open or t_ml is not None)
        civ_no_ceiling = t_ch is None and (civ_open or t_cl is not None)
        total_lo = totals.get("grand_low")
        total_hi = totals.get("grand_high")

        # Residual closure: a missing class ceiling is closed mechanically by
        # the grand-total residual.  It requires a grand ceiling and a real
        # ceiling on the other class; otherwise the closure would be vacuous
        # and the war stays flagged incomplete.
        grand_hi = float(total_hi) if total_hi is not None else None
        mil_incomplete = civ_incomplete = False
        if mil_no_ceiling:
            if grand_hi is not None and not civ_no_ceiling:
                mil_hi = max(mil_lo, grand_hi - civ_lo)
            else:
                mil_incomplete = True
        if civ_no_ceiling:
            if grand_hi is not None and not mil_no_ceiling:
                civ_hi = max(civ_lo, grand_hi - mil_lo)
            else:
                civ_incomplete = True
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
            mil_incomplete=mil_incomplete,
            civ_incomplete=civ_incomplete,
        ))
    return sorted(wars, key=lambda w: -w.total_mid)


def row(w: War, long_name: bool = False) -> str:
    name = latex_escape(w.name)
    if not long_name and len(name) > 42:
        name = name[:39] + r"\ldots{}"
    grade = effective_grade(w)
    diag = "source harmonisation" if (w.mil_incomplete and w.civ_incomplete) else diagnostic(w)
    if w.mil_incomplete and w.civ_incomplete:
        civ_share = q_range = "n.a."
    elif w.mil_incomplete:
        civ_share = f"$\\le${pct(w.civ_share_hi)}"
        q_range = f"$\\ge${pct(w.q_lo)}"
    elif w.civ_incomplete:
        civ_share = f"$\\ge${pct(w.civ_share_lo)}"
        q_range = f"$\\le${pct(w.q_hi)}"
    else:
        civ_share = pct_range(w.civ_share_lo, w.civ_share_hi)
        q_range = pct_range(w.q_lo, w.q_hi)
    return (
        f"{name} & {w.period} & {count_fmt(w.total_mid)} "
        f"& {civ_share} "
        f"& {q_range} "
        f"& {pct(w.indirect_share)} "
        f"& {latex_escape(diag)} "
        f"& {grade} \\\\"
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
    grades = {g: sum(1 for w in wars if effective_grade(w) == g) for g in "ABCD"}
    n_na = sum(1 for w in wars if effective_grade(w) == "n.a.")
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
        r"The $q$ interval is the corresponding combatant share. "
        r"\emph{Missing figures are treated explicitly rather than as zeros}: when the sources "
        r"state no ceiling for one casualty class, the ceiling is either taken from the curated "
        r"per-conflict totals (with the reasoning recorded in the data file's notes) or closed "
        r"mechanically by the grand-total residual---the total-death ceiling minus the other "
        r"class's floor. Should a conflict identify only a one-sided bound, it is marked "
        r"$\le x\%$ (combatant deaths incompletely attributed) or $\ge x\%$ (civilian deaths "
        r"incompletely attributed), or \emph{n.a.} if neither class is attributable; after "
        r"curation no conflict in the current dataset requires these marks. "
        r"The \emph{binding relation} column is a triage label: "
        r"\texttt{manpower budget} means the population/combatant-stock bound is the first place to look; "
        r"\texttt{identified-deaths sample} means sex-age microdata would likely dominate; "
        r"\texttt{excess-mortality survey} means direct attribution is less informative than survey-based excess deaths; "
        r"\texttt{source harmonisation} marks conflicts where definitions, overlap, or missing high bounds dominate.",
        "",
        r"\paragraph{Confidence grade.} A: total-death range ratio $<2$ and civilian-share width $<15$ percentage points. "
        r"B: either ratio $<4$ or width $<30$pp. C: either ratio $<8$ or width $<45$pp. D: wider than that. "
        r"Conflicts with a one-sided share bound are graded on the widened (identified) interval, "
        r"so an incompletely attributed class demotes the grade rather than flattering it. "
        r"These grades score internal numerical precision, not moral seriousness or historical importance.",
        "",
        r"\section{Dataset summary}",
        rf"The dataset contains {len(wars)} conflicts. Confidence grades: "
        rf"A={grades['A']}, B={grades['B']}, C={grades['C']}, D={grades['D']}, "
        rf"not attributable={n_na}. "
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
        r"The table below lists the primary sources used to construct the casualty bounds for each "
        r"conflict: official records, court and truth-commission documents, UN and NGO reports, "
        r"scholarly monographs, and press of record. Tertiary reference works consulted only for "
        r"orientation are omitted here; the full source list for every conflict, with URLs, is in "
        r"the per-conflict JSON files in the companion repository, which also contain detailed "
        r"side-by-side accounting, indirect-death breakdowns, and specific atrocity figures.",
        r"\begin{scriptsize}",
        r"\begin{longtable}{p{3.5cm} p{12.5cm}}",
        r"\toprule",
        r"Conflict & Primary sources \\",
        r"\midrule",
        r"\endhead",
    ]

    # Tertiary aggregators are dropped whenever the conflict has at least two
    # primary sources left; long lists are capped with an explicit pointer to
    # the repository rather than truncated silently.
    tertiary = {
        "wikipedia", "new world encyclopedia", "history.com", "onwar",
        "grokipedia", "thoughtco", "britannica.com",
    }
    max_listed = 8

    def is_tertiary(pub: str) -> bool:
        p = pub.lower()
        return any(t in p for t in tertiary)

    for w in wars:
        if not w.sources:
            continue
        seen: set[tuple[str, str]] = set()
        primary, rest = [], []
        for s in w.sources:
            title = (s.get("title") or "").strip()
            pub = (s.get("publisher") or "").strip()
            key = (title.lower(), pub.lower())
            if not (title or pub) or key in seen:
                continue
            seen.add(key)
            (rest if is_tertiary(pub) else primary).append((title, pub))
        kept = primary if len(primary) >= 2 else primary + rest
        n_omitted = len(seen) - min(len(kept), max_listed)
        kept = kept[:max_listed]

        src_texts = []
        for title, pub in kept:
            if title and pub:
                src_texts.append(f"\\emph{{{latex_escape(title)}}} ({latex_escape(pub)})")
            elif title:
                src_texts.append(f"\\emph{{{latex_escape(title)}}}")
            else:
                src_texts.append(latex_escape(pub))
        if not src_texts:
            continue
        tail = f"; and {n_omitted} further sources in the repository" if n_omitted > 0 else ""
        lines.append(f"{latex_escape(w.name)} & {'; '.join(src_texts)}{tail} \\\\")
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
