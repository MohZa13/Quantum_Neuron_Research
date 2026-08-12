# docs/ — the knowledge base

Entry point for the whole repo is `../AGENTS.md`. This file indexes `docs/`.

## The framework (maintained, authoritative)

| File | What it is | Update when |
|---|---|---|
| [`ORIENTATION.md`](ORIENTATION.md) | Plain-language project brief for a cold reader. What we are building, why, and where the two halves stand. | The project's direction or state materially changes |
| [`REPO_MAP.md`](REPO_MAP.md) | Every file in the repo, one line each, grouped by directory. | Any file is added, removed, or renamed |
| [`GLOSSARY.md`](GLOSSARY.md) | Domain terms — quantum chemistry, quantum ML, and this repo's own jargon. | A term appears that a newcomer could not look up |
| [`DATA_CATALOG.md`](DATA_CATALOG.md) | Every dataset and artifact: path, provenance, schema, status, regeneration cost. | Any dataset is produced, consumed, invalidated, or deleted |
| [`INVARIANTS.md`](INVARIANTS.md) | The do-not-break list, each with evidence and a verification command. | A new trap is discovered, or an invariant is verified/retired |
| [`QUANTUM_NEURON.md`](QUANTUM_NEURON.md) | **The focus document.** The model, its exact hypothesis class, the coherence-label program, and the roadmap. | The model, its labels, or its theory advance |
| [`HYBRID_BACKPROP.md`](HYBRID_BACKPROP.md) | **The derivation.** Backpropagation from a classical layer into a quantum neuron — the training rule the source paper leaves open, and the specification for [`qnn/`](../qnn/README.md). | The network architecture, its gradients, or its numerics change |
| [`RESEARCH_LOG.md`](RESEARCH_LOG.md) | Dated, append-only ledger of findings. The project's memory. | Anything true is learned — positive or negative |
| [`OPEN_QUESTIONS.md`](OPEN_QUESTIONS.md) | Prioritized live research agenda, each question with a decisive test. | A question is raised, answered, or reprioritized |
| [`DECISIONS.md`](DECISIONS.md) | ADR log: choices, alternatives, rationale, status. | A choice is made that a future reader would question |
| [`WORKFLOWS.md`](WORKFLOWS.md) | Runbooks: exact commands for every recurring task. | A procedure is established or changes |
| [`AGENT_PROTOCOL.md`](AGENT_PROTOCOL.md) | How to work in this repo: conventions, update rules, checklists. | The working conventions change |
| [`templates/`](templates/) | Copy-paste stubs for experiments, findings, and decisions. | — |

## Topic reports (point-in-time, not continuously maintained)

These are snapshots. Each carries its own status header; trust
`RESEARCH_LOG.md` and `DATA_CATALOG.md` over them where they disagree.

| File | Topic | Status |
|---|---|---|
| [`OMOL25_ASSESSMENT.md`](OMOL25_ASSESSMENT.md) | **Should we move to OMol25?** The dataset's verified contents, ten second-quantization labels tested and failed, the basis-invariance audit, the admission criterion QH9 cannot meet, engineering requirements, cost, and a staged decisive test | Written 2026-08-06; the evidence is measured, the OMol25 facts are cited to arXiv:2505.08762v2 |
| [`thermal_pipeline_report.md`](thermal_pipeline_report.md) | Plain-language report on the QH9 → thermal pipeline | Written 2026-07-15 at the 50-molecule stage; superseded numbers flagged inline |
| [`classifier_optimization.md`](classifier_optimization.md) | Classifier bottleneck analysis and optimization history | Current |
| [`scaling_comparison.md`](scaling_comparison.md) | Measured scaling: original vs optimized classifier | Current (machine-specific timings) |
| [`paper_comparison_guide.md`](paper_comparison_guide.md) | How to run and read the paper-style comparison figures | Current |
| [`pennylane_quick_start.md`](pennylane_quick_start.md) | Early PennyLane integration notes | **Largely superseded** — its "next steps" are done or deliberately skipped |

## Assets

- `fig_8.png` — the source paper's Fig. 8, digitized by
  `figures/digitize_fig8_classical.py` into `results/digitized/`.

## Reading order for a cold start

1. `../AGENTS.md` — map and routes
2. `ORIENTATION.md` — what and why
3. `GLOSSARY.md` — vocabulary (skim; return as needed)
4. `QUANTUM_NEURON.md` — the actual research problem
5. `OPEN_QUESTIONS.md` — what to do next

Everything else is reference, consulted on demand.
