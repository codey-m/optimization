# exp-a-spiel, patched for Lab 4

[exp-a-spiel](https://github.com/gabrfarina/exp-a-spiel) is Gabriele Farina's C++ package, with Python bindings,
for fast gradients, exploitability and tabular solvers in Dark Hex and phantom tic-tac-toe. It is used here under
its MIT license; `LICENSE` is the upstream license, unchanged.

| | |
|---|---|
| Upstream commit | `9412882` (2025-02-20) |
| Patch | `corner_pttt.patch` |
| Built by | `.github/workflows/build-eas-wheel.yml`, on a tag `eas-v*` |

What the patch changes:

- **`CornerAbruptPttt`**: abrupt phantom tic-tac-toe with a fixed, commonly known opening. Each player's first
  move is pre-placed, and a player may not probe a cell the opponent is known to hold. `set_pttt_opening` sets the
  opening before a traverser is built; Lab 4 uses player 0 on cell 4 (center) and player 1 on cell 0 (corner).
- **Quiet logging**: the package's progress messages print only when the environment variable `EAS_VERBOSE` is set.
- **Portable build flags**: `-march=x86-64-v2` in place of `-march=native`, so a wheel built on one machine runs on
  any learner runtime.
- **Headers-only Boost for CI**: when `BOOST_INCLUDEDIR` is set, CMake uses those headers directly instead of looking
  for Boost's CMake package; otherwise the upstream lookup is unchanged.
- **Version** `0.0.1.post1`, so the patched build has its own name. (A `+` local version would be renamed by GitHub on a release asset, and pip would then refuse the file.)

`smoke_test.py` is the test CI runs against every wheel: the lab's game has 46,824 and 29,435 information sets and
a uniform NashConv of 1.2720238.
