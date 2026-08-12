# Template — experiment record

For a planned run: fill §1–3 **before** starting, §4–6 after. Keep the file
next to its output (e.g. `results/<name>.md` beside `results/<name>.h5`), and
add a row to `docs/DATA_CATALOG.md`.

Writing the hypothesis and the falsification criterion *before* the run is the
point. It is what stops a confounded result from being reported as a finding.

---

# Experiment: [name]

**Date started:** YYYY-MM-DD · **Status:** planned | running | complete | abandoned

## 1. Question

[Which `OPEN_QUESTIONS.md` entry does this address? If none, why is it worth
the compute?]

## 2. Hypothesis and falsification

**Expected:** [what you think will happen, quantitatively if possible]

**Would falsify it:** [the concrete outcome that means "no". Write this now,
not after seeing the data.]

**Would confound it:** [what could produce a positive result for the wrong
reason. For classification results, check `AGENT_PROTOCOL.md` §8.]

## 3. Setup

**Command:**
```bash
[exact command, including every flag]
```

**Inputs:** [source files, molecule selection and how it was made]
**Parameters:** [active space, kT ladder, solver, keep-cap, cutoff, seeds]
**Outputs:** [paths to be written]
**Estimated cost:** [wall time, disk, RAM]

## 4. Execution

**Started / finished:** [timestamps] · **Actual cost:** [wall, disk]
**Interruptions / resumes:** [what happened]
**Warnings in the log:** [especially `cap_hit`, unconverged Davidson, skips —
with counts and magnitudes]

## 5. Results

[Numbers, with units and spread. Truncation errors alongside anything derived
from a truncated ensemble.]

## 6. Conclusion

**Answer:** [confirmed | falsified | inconclusive — and why]
**Caveats:** [confounds that survive]
**Next:** [what this makes possible or necessary]

→ Promote the finding to `docs/RESEARCH_LOG.md`; update
`docs/OPEN_QUESTIONS.md` and `docs/DATA_CATALOG.md`.
