# Per-journal submission packages

Each subfolder contains a journal-ready draft (`main.tex` + `main.pdf`) and a cover letter (`cover_letter.tex` + `cover_letter.pdf`) for one target journal. All five share the master content in `../content/*.tex` and the master BibTeX in `../content/refs.bib`; the only thing that changes between journals is the preamble, the abstract format, and the cover letter.

**Author (single, corresponding):** David H. Silver, Independent researcher, Poughkeepsie, NY 12603, USA. <silverdavi@gmail.com> · ORCID [0000-0002-3071-304X](https://orcid.org/0000-0002-3071-304X).

| Folder | Journal | Tier | Length budget | Abstract format | Citation style | Notes |
|---|---|---|---|---|---|---|
| [`pnas/`](pnas/README.md) | *PNAS* (Direct Submission, Social Sciences) | 1 (high impact) | 6 pp standard / 12 pp max | + 120-word Significance Statement | Numerical (`pnas.bst`) | First decision in 2–4 weeks; ~14% acceptance. |
| [`jcr/`](jcr/README.md) | *Journal of Conflict Resolution* (SAGE) | 1 (field flagship) | 8–12k words | Plain | Harvard (`SageH.bst`) | Spagat 2009 precedent; Vesco 2026 in same volume. |
| [`aoas/`](aoas/README.md) | *Annals of Applied Statistics* (IMS) | 1 (methodology) | 25–40 typeset pp | Plain + MSC2020 | Author-year (`imsart-nameyear.bst`) | Methodology-first; reproducibility statement required. |
| [`jpr/`](jpr/README.md) | *Journal of Peace Research* (OUP) | 2 | ~10k words inc. refs | Plain | Author-year | OUP transition complete (online-only since Jan 2026). |
| [`pophm/`](pophm/README.md) | *Population Health Metrics* (BMC, open access) | 2 | No hard limit | Structured (Background/Methods/Results/Conclusions) | Vancouver (numerical) | Complementary to Gómez-Ugarte et al. 2025 in same journal. |

## How the master ↔ children architecture works

```
paper/
├── paper.tex                # MASTER (compiles a generic 12-pp draft)
├── content/
│   ├── abstract.tex
│   ├── 01_intro.tex
│   ├── ...
│   ├── 08_discussion.tex
│   └── refs.bib             # MASTER BibTeX, 47 entries
├── figures/                 # MASTER figures (shared)
└── submissions/
    ├── pnas/   main.tex     ── \input{../../content/...}, journal preamble, +Sig statement
    ├── jcr/    main.tex     ── \input{../../content/...}, SAGE preamble, +keywords
    ├── aoas/   main.tex     ── \input{../../content/...}, IMS preamble, +MSC codes
    ├── jpr/    main.tex     ── \input{../../content/...}, OUP preamble, +keywords
    └── pophm/  main.tex     ── \input{../../content/...}, BMC preamble, +structured abstract
```

**Edit content in one place** (`../content/*.tex`); every journal version picks the change up on the next compile.

## Build everything

```bash
# from paper/
pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
for d in submissions/*/; do
  ( cd "$d" \
    && pdflatex -interaction=nonstopmode main \
    && bibtex main \
    && pdflatex -interaction=nonstopmode main \
    && pdflatex -interaction=nonstopmode main \
    && pdflatex -interaction=nonstopmode cover_letter )
done
```

## Before submitting to a real journal

Each `README.md` inside each subfolder lists journal-specific requirements (class file, portal, anonymisation, replication-data deposit). The current drafts use the standard `article` class so the master tree is portable; swapping in the journal's official `.cls` file is a localised preamble change, the body content does not need to move.

## Drop-in citation customisation

If a target journal wants extra citations (e.g., JPR may want more UCDP/ACLED references; Pop Health Metrics may want more *Lancet* / *BMJ Global Health* references), add them to `../content/refs.bib` and cite from the journal-specific `main.tex` only. The master BibTeX file is the union of everything cited anywhere; LaTeX will silently ignore entries not cited in a given main file.
