# ML numeric variation

**Situation:** Two GPU runs differ slightly, while paired metrics remain within the planned tolerance and the ranking is unchanged.

**Overreaction:** Add artifact hashes, force byte-identical tensors, rerun every stage, and treat ordinary kernel variation as corruption.

**General response:** Record seeds and the relevant environment, report variation across planned runs, and judge the claim with tolerances or confidence intervals.

**Escalate only when:** variation exceeds the declared tolerance, changes the decision, or indicates a real data, code, or environment mismatch.
