# Reuse before rebuilding

**Situation:** The preferred parser or converter is unavailable in the current environment.

**Overreaction:** Immediately write a custom parser, wrapper, validator, and fallback chain.

**General response:** Search the repository, declared dependencies, standard library, and official toolchain. Reuse a mature implementation that satisfies the contract; custom-build only the missing edge.

**Rule:** Tool unavailability is a discovery prompt, not evidence that the feature must be reimplemented.
