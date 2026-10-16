# Implications memo — Archive Genocide event graph vs the POPHM Gaza claims

**Status: bridge, not rewrite.** No number in the POPHM manuscript, the
`gaza_sim/` posterior, or `data/per_war/israel_gaza_war_2023.json` is changed
by this memo. This module analyses an independent primary-footage event graph
and maps what it can and cannot say about each claim the paper already makes.

**Source.** [Gaza Node Map / lab](https://archivegenocide.com/lab/) event
graph, build `f76bd2fb` (pinned in `BUILD_ID` and
`ingest/PINNED_SOURCE.json`), 92,588 events / 131,387 clip-metadata rows.
Data © Archive Genocide, licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); event-graph CSVs
only, no media downloaded. GPG signature (key `9B4D575499AFA53C`) and all
SHA256 sums verified at ingest.

**Dataset constraints, taken as method (not footnotes).** An event is a
*cluster of footage*, so event counts are **upper bounds** on distinct
incidents. `killed_min`/`killed_max` are caption-stated cluster figures and
are **never summed** anywhere in this module (a unit test enforces this).
Coordinates are gazetteer **place anchors**, not strike locations. Most dates
are publication dates; primary analyses use `date_source == stated` (10,690
events), with a separate `post_proxy` sensitivity that is never mixed in.
`verification_tier` measures corroboration *inside this archive*, not truth —
tier D means single-source here, not false.

## Claim-by-claim map

### 1. Identification of the combatant share q — unsupported here, but the named denominator is real

Nothing in this dataset identifies q, and this memo makes no attempt to.
What the name-overlap analysis (`aggregates/name_overlap.json`) does show:
of 8,627 people named in primary footage captions, 1,276 (14.8%) are
unambiguous matches to the Ministry of Health identified-dead register
(140 more are ambiguous leads; 83.6% unmatched), and 1,085 events name a
register-matched person. The identities appearing independently in primary
captions are the same identities populating the MoH identified list. This
supports the paper's use of that list as a **real named denominator** for the
identification argument — it does not estimate q and does not replace a
household survey. Counts only; no names appear in any committed artefact.

### 2. IDF 17–25k combatant claim — remains a demographic question

The footage graph cannot arbitrate a combatant count. What it shows,
**under selection bias** (people film what they film), is that documented
events are heavily civilian-context: 13,924 events involve a struck facility,
including 4,607 hospital/medical, 1,879 school/university, and 2,379
shelter/camp events; 9,275 events mention children, 5,182 journalists, and
4,102 medics (`aggregates/protected_sites.json`). This is documentation
intensity, not a legal finding and not a demographic sample. The paper's
feasibility analysis of the 17–25k claim stays exactly where it was: a
question about demographic composition of the identified dead.

### 3. Spatial structure — flat-per-capita documentation is empirically rejected

The paper's simulator (`gaza_sim/01_simulator.py`) invents spatial targeting
concentration through its α/γ parameters over five homogeneous governorate
cells, and its report flags this as a limit. The stated-date, mapped subset
(3,465 of 4,544 Gaza-bbox events assigned to a governorate) rejects a
population-proportional null: chi-square 67.2 on 4 df, p ≈ 9×10⁻¹⁴
(`aggregates/spatial.json`). Governorate-level deviations are moderate
(e.g. Deir al-Balah 18.5% observed vs 14.3% population share); the strong
result is **concentration at named-place grain**: the top decile of the 79
place anchors carries 70.8% of events, Gini 0.82. So documentation is
spatially peaked, not flat per capita — consistent with the simulator's
choice to model concentration at all, and evidence against a homogeneous-cell
world. Two caveats bind: place anchors are gazetteer centroids, **not strike
locations** (this is named-place documentation intensity), and the
`post_proxy` sensitivity (22,945 assigned events; chi-square 1,758; Gaza
governorate at 45.6% observed) is reported separately because
publication-date data skew toward where publishers are, not where strikes are.

### 4. μ_M (civilian working-age-male exposure) — not identified by captions

Captions are not a demographic sample of the dead. Nothing here estimates
μ, and this module does not pretend otherwise. The paper's μ range remains
sourced where it was.

### 5. Indirect deaths — qualitative only

Aftermath footage and the healthcare-facility strike counts above are
qualitative context for the indirect-deaths bucket. No indirect-death
quantity is derived from this dataset.

## Corroboration structure (the "conflicting sources" theme, from the other side)

From `aggregates/corroboration.json`: verification tiers are D 90.5%,
B 5.7%, C 2.2%, A 1.6%. Only 2.8% of events have ≥2 distinct pieces of
footage, and 6.1% appear in ≥2 archives — most events are filmed once and
reposted. At the other extreme there is a hard core of 255 events present in
5+ archives with 5+ distinct footage. `figures_disputed` is rare (0.75%).
Event–event links split into 49,030 established structural links (shared
entity/person, same day, repeat target) vs 10,262 `possible_same_event`
candidate-duplicate flags — a direct, independent illustration of the
duplication problem the paper's aggregation framework is built around.
Tier D is single-source *within this archive*, not false.

## Incident-size distribution (never a toll)

Among stated-date, mapped, non-cumulative rows with a caption figure
(`aggregates/incident_size.json`): median `killed_min` is 2 and about 25% of
clusters state ≥10 killed (heavy-tailed; the p99 is in the hundreds).
Across all 12,736 non-cumulative rows, 24 mis-filed war-to-date totals were
flagged (max 514,000) and excluded from the cleaned distribution. These are
caption-stated cluster sizes; they are never summed, and the presence of a
`report_day` means only that a daily MoH report exists for that date, not
event-level confirmation.

## Spot checks (tier A, high corroboration)

Five tier-A events with the highest archive × footage corroboration, opened
in the archive's viewer as it requests (event ids only; no victim names):

- https://archivegenocide.com/lab/#focus=ev-e1f9dfecb0 — Gaza City, 13 archives, 67 distinct footage
- https://archivegenocide.com/lab/#focus=ev-72eead79bd — Kamal Adwan Hospital strike, 11 archives, 115 distinct footage
- https://archivegenocide.com/lab/#focus=ev-026c5522f5 — Nasser Hospital strike, 10 archives, 124 distinct footage
- https://archivegenocide.com/lab/#focus=ev-e231241364 — Dar Al-Arqam School strike, 10 archives, 77 distinct footage
- https://archivegenocide.com/lab/#focus=ev-0a0af8ac60 — Al-Shati Refugee Camp strike, 10 archives, 58 distinct footage

## Bottom line

The event graph strengthens the paper from the side: it shows the MoH
identified list is populated by independently filmed identities, it rejects
flat-per-capita spatial documentation, and it quantifies how rarely incidents
are independently corroborated versus reposted. It identifies neither q nor
μ, produces no death toll, and changes **no number** in the POPHM manuscript,
the `gaza_sim` posterior, or the per-war JSON.
