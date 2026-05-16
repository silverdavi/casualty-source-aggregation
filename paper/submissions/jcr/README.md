# JCR submission package

**Target:** *Journal of Conflict Resolution* (SAGE), Research Article.

## Files

| File | Purpose |
|---|---|
| `main.tex` | Manuscript (built on shared `../../content/*.tex` + `../../content/refs.bib`). |
| `title_page.tex` | Detached, un-blinded title page (SAGE convention; corresponding author + bio note). |
| `cover_letter.tex` | Cover letter. |
| `README.md` | This file. |

## Build

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
pdflatex cover_letter
pdflatex title_page
```

## Submission checklist (JCR / SAGE)

1. **Class file.** Replace the `article` preamble with the SAGE class:
   - Download `sagej.cls` and `SageH.bst` from the [SAGE LaTeX page](https://uk.sagepub.com/en-gb/eur/manuscript-submission-guidelines).
   - Use `\documentclass[Crown,times]{sagej}` per the SAGE skeleton; keep `\input{...}` and `\bibliography{...}` lines.
   - Switch `\bibliographystyle` to `SageH` (Harvard).
2. **Length.** JCR Research Articles are typically 8,000–12,000 words; supplements unlimited. Current main text is ~7,500 words — comfortably within budget.
3. **Submission portal.** ScholarOne Manuscripts (`mc.manuscriptcentral.com/jcr`).
4. **Anonymous main file.** SAGE asks for a separate title page for double-blind review; the un-blinded title page is in `title_page.tex` (compiles to `title_page.pdf`). For peer-review submission, replace the `\author{...}` block in `main.tex` with `\author{Anonymous}` (instructions are in a comment at that line); restore for camera-ready / acceptance.
5. **Keywords.** Already in `main.tex`: casualty estimation; civilian victimisation; partial identification; robust Bayesian inference; source aggregation; Gaza; under-fudgible relations.

## Key citations to highlight in cover letter

- Spagat et al. 2009 JCR ("Estimating war deaths: an arena of contestation") — the canonical precedent.
- Eck & Hultman 2007 JPR (one-sided violence data).
- Lacina & Gleditsch 2005 EJP (battle-deaths dataset).
- Vesco et al. 2026 JCR (UCDP underreporting; concurrent issue).
- Radford et al. 2023 PNAS (Bayesian conflict losses, complementary).

## Possible reviewers (no conflict)

Michael Spagat, Patrick Ball, Megan Price, Lisa Hultman, Bethany Lacina, Diego Alburez-Gutierrez.
