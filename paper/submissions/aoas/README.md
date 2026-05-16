# AOAS submission package

**Target:** *Annals of Applied Statistics* (Institute of Mathematical Statistics).

## Files

| File | Purpose |
|---|---|
| `main.tex` | Manuscript (built on shared `../../content/*.tex` + `../../content/refs.bib`). |
| `cover_letter.tex` | Cover letter. |
| `README.md` | This file. |

## Build

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
pdflatex cover_letter
```

## Submission checklist (AOAS / IMS)

1. **Class file.** Replace `article` with the IMS `imsart` class:
   - Download from [imstat.org/imsart](https://www.imstat.org/imsart/).
   - Use `\documentclass[aoas]{imsart}` per the imsart guide.
   - Use `\bibliographystyle{imsart-nameyear}`.
2. **Length.** AOAS does not have a hard word limit; methodology papers typically run 25–40 typeset pages. Move proofs of regularity conditions and the 81-conflict tables to the supplementary material.
3. **MSC2020 codes.** Already in `main.tex`: 62F15 (Bayesian inference), 62P25 (statistics in social sciences), 62G35 (robustness), 62F35 (robustness and adaptive methods).
4. **Submission portal.** Editorial Manager (`https://www.e-publications.org/ims/submission/AOAS/`).
5. **Reproducibility.** AOAS *requires* a reproducibility statement and code/data deposit. Add a final paragraph naming the public repository.

## Key citations to highlight in cover letter

- Manski 2003, 2007, Molinari 2020 (partial identification — methodological backbone).
- Berger & Berliner 1986; Gagnon 2023; Di Noia–Ruggeri–Mira 2025 ($\varepsilon$-contamination).
- Lum, Price, Banks 2013 The American Statistician (closest applied cousin).
- Romano & Shaikh 2010; Horowitz & Manski 1995 (inference for identified sets).
- Radford et al. 2023 PNAS (contemporary Bayesian source-aggregation).
- Fagan et al. 2020 JRSS-A (battle-deaths change-points).

## Possible Associate Editors / reviewers (no conflict)

Charles Manski, Aaron Schein, Tirthankar Dasgupta, Susan Murphy; Patrick Ball, Megan Price, Michael Spagat, Diego Alburez-Gutierrez, Francesca Molinari.
