# Test overengineering

**Situation:** A small deterministic transformation needs regression coverage.

**Overreaction:** Mock every internal call, add factories and fixtures, assert call order, and chase coverage across unreachable branches.

**General response:** Test representative input and output, one meaningful boundary, and any reproduced failure. Mock only a real external boundary.

**Rule:** Test complexity should track behavioral risk, not the number of implementation details.
