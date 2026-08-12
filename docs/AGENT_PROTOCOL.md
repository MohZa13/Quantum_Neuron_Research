# Agent protocol — how to work in this repository

*Conventions and maintenance rules. The goal is that the **next** cold start is
cheaper than this one.*

---

## 1. Starting a session

1. Read `AGENTS.md` (map + routes). `CLAUDE.md` is already loaded.
2. Pick your route from the table there. **Read only those files.**
3. Before proposing work, check `OPEN_QUESTIONS.md` — the priority is usually
   already decided, and Q1 blocks most of the rest.
4. Before claiming anything is new, check `RESEARCH_LOG.md`. Several ideas here
   have already been measured, including ones that failed.

**Do not explore by `ls`/`find`.** `docs/REPO_MAP.md` catalogs every file and
`docs/DATA_CATALOG.md` every artifact. If something is missing from them, that
is a bug in the docs — fix it as part of your task.

---

## 2. Ground rules

**Verify before asserting.** This project has already been burned once by
numbers that looked fine and were not ([`INVARIANTS.md`](INVARIANTS.md) I1).
If you state a count, a size, or a result, run the command that produces it.

**Distinguish measured from assumed.** In every doc: measured numbers cite the
artifact or command; hypotheses say "expected". Two claims in this repo were
stated as expectations and later overturned by measurement — the MPS ordering,
and the classical FCIM's ability to fit a mixed target. Both were caught only
because someone measured.

**Truncation and error bars are never optional.** If you produce a number from
a truncated ensemble, carry `truncation_error` and `cap_hit` alongside it.

**Prefer the eigenblock formulation.** If your analysis needs a dense ρ, you
have probably not found the right contraction yet
([`INVARIANTS.md`](INVARIANTS.md) I3).

**Match the surrounding style.** The code here has a consistent voice: dense
explanatory docstrings that state *why*, not *what*; comments that record
evidence and dates; assertions that encode invariants. Follow it.

**Negative results are results.** "We measured X and it does not work" belongs
in `RESEARCH_LOG.md`. Losing that is how a project repeats itself.

---

## 3. What to update, and when

| You did this | Update |
|---|---|
| Learned something true about the science (incl. negative) | `RESEARCH_LOG.md` — new dated entry at the top |
| Made a non-forced choice | `DECISIONS.md` — new ADR |
| Created / deleted / invalidated a file or dataset | `DATA_CATALOG.md` **and** `REPO_MAP.md` |
| Added or removed a source file | `REPO_MAP.md`, and the directory's own `README.md` |
| Answered a research question | Move it to `OPEN_QUESTIONS.md` → *Answered*, link the log entry |
| Raised a new question | `OPEN_QUESTIONS.md`, with a decisive test and a priority |
| Found a trap that causes silent wrong answers | `INVARIANTS.md`, with evidence and a verification command |
| Established a repeatable procedure | `WORKFLOWS.md` |
| Introduced a term a newcomer could not look up | `GLOSSARY.md` |
| Advanced the model, its labels, or its theory | `QUANTUM_NEURON.md` |
| Found a stale claim anywhere | Fix it, and log the correction in `RESEARCH_LOG.md` |

**Append, do not overwrite.** `RESEARCH_LOG.md` and `DECISIONS.md` are
append-only. To supersede an entry, edit the old one to say so and link
forward. Rewriting a "current status" paragraph destroys precisely the history
a future reader needs.

---

## 4. Writing style for these docs

Written for an agent with **zero context** and for a reader without deep domain
knowledge. Concretely:

- **Lead with the claim**, then the evidence. Not the other way round.
- **Numbers carry units and provenance.** "median 192 s/molecule (production
  run log)" — not "fast".
- **Say what would falsify it.** Every open question names a decisive test.
- **Mark uncertainty explicitly.** "measured", "expected", "unverified" are
  different words. Use the right one.
- **Cross-link.** A reader landing mid-document should be one hop from context.
- **No filler.** If a section has nothing to say, delete the section.

---

## 5. Before finishing a task

```bash
.venv/bin/python -m pytest tests/          # 156 expected
git status --short                         # know what you changed
```

Then ask, in order:

1. Did I verify every number I stated?
2. Did I update the docs my change invalidates (table in §3)?
3. Did I create an artifact without provenance? (If yes, add it to
   `DATA_CATALOG.md` — or write the script, which is better.)
4. Did I learn something the next agent would want, and is it written down?
5. Did I contradict an existing doc? If so, one of them is wrong — resolve it,
   do not leave both.

---

## 6. Repo-specific hazards

Read [`INVARIANTS.md`](INVARIANTS.md) in full before touching `qthermal/`. The
short list of things that fail *silently*:

- AO reordering raw DB Hamiltonians (I1) — plausible numbers, wrong physics
- Running SCF (I2) — a different physical object from the source dataset
- Reordering notebook cells (I6) — looks like a numerical failure
- Assuming `evals` exists (I5) — works on dense runs, breaks on Krylov ones
- Mixing wire orderings across files (I12) — no error, meaningless comparison

And two operational ones:

- **`results/*.h5` are gitignored and local.** Never assume a file is present;
  check, and regenerate per `DATA_CATALOG.md` if not.
- **The 45 GB production file is expensive** (13.3 h to rebuild). Read it, do
  not overwrite it. Write new runs to new paths.

---

## 7. Long-running work

Pipeline runs take hours. Design for interruption:

- **Every stage is resumable** — `qthermal.run`, `qthermal.encode_run`, and
  `screen_conjugation.py` all checkpoint. Rerunning the identical command is
  always the correct recovery.
- **Log to a file** (`2>&1 | tee results/<name>.log`) and catalog the log. Run
  logs are how we reconstruct what a dataset actually contains — the cap
  warnings in `qh9_conjugated_top45.log` are the only record of which blocks
  hit the storage ceiling.
- **Record the exact command** in `DATA_CATALOG.md` next to its output. A
  dataset whose generating command is lost is halfway to being an orphan.

---

## 8. Scope discipline

The interesting failure mode here is not doing too little — it is producing a
result that looks good and is confounded. Before reporting any classification
result, check:

1. Could a classical descriptor model get the same accuracy?
   (`OPEN_QUESTIONS.md` Q4)
2. Does the Z-only ablation land at chance? (Q2)
3. Is the label sensitive to active-space size? (Q3)

A result that has not passed these is a preliminary observation, and should be
reported as one.
