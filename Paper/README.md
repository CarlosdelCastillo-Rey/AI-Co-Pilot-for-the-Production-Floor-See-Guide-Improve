# Academic paper (LaTeX)

IMRaD manuscript for the **VisionOps** notebook research track: per-person HAR with V-JEPA~2 on InHARD.

## Structure

| File | Role |
|------|------|
| `main.tex` | Document entry point |
| `preamble.sty` | Packages, typography (see `how-to-create-a-paper.md`) |
| `sections/01-introduction.tex` … `06-conclusion.tex` | IMRaD body |
| `references.bib` | Bibliography (InHARD, V-JEPA, YOLO, ByteTrack) |
| `how-to-create-a-paper.md` | Rhetorical and LaTeX style guide |

Figures are loaded from notebook outputs:

- `../notebooks/outputs/inhard_eda/` — EDA charts (step 01b)
- `../notebooks/outputs/har_analysis/2026-06-07_all14_5each/` — pilot metrics (step 06)

After a new training run, update `\graphicspath` in `preamble.sty` or add a dated analysis folder.

## Build

```bash
cd Paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or upload `Paper/` to [Overleaf](https://www.overleaf.com) and set the main document to `main.tex`.

## Updating results

1. Run `notebooks/06_Model_and_Session_Analysis.ipynb` after each checkpoint.
2. Copy key numbers from `outputs/har_analysis/*/REPORT.md` into `sections/04-results.tex`.
3. Refresh the abstract last (per `how-to-create-a-paper.md`).

## Style checklist

- No first/second person (`I`, `we`, `you`) in body text
- Past tense in Methods and Results; hedged claims in Discussion
- Figures referenced before interpretation; results separated from discussion
