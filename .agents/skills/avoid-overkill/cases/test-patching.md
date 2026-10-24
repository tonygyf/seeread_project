# Test patching and fallbacks

**Situation:** A behavioral test fails because a required executable is absent.

**Bad patch:** Silently choose another interpreter, catch every exception, or replace the behavioral assertion with a source-text check.

**General response:** Preserve the public contract. Fail clearly, or support an explicit documented override. Keep the behavioral test and fix the environment boundary or implementation.

**Rule:** A passing test must still exercise the behavior the user depends on.
