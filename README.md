# Optimization learner materials

Public learner notebooks for the Optimization course sequence, 6.7205.1x to 6.7205.4x. The current build holds
the Module 4 lab described below.

## Start here

Open notebooks from their course pages and make a copy in Google Drive before working. The setup cell installs
the lab's package and downloads its checked support files, so learners do not need to clone this repository.

The course pages link to notebooks by their path in this repository, `codey-m/optimization` on the `main` branch,
so renaming a notebook, moving it, or renaming the repository breaks its Colab button.

## Current notebook build

| Notebook | Learner work | Course location |
|---|---|---|
| [`labs/lab4_exploitability.ipynb`](labs/lab4_exploitability.ipynb) | Write regret matching, the sequence form, counterfactual values and a best response; run CFR and its CFR+ refinements on a fixed-opening phantom tic-tac-toe; compare a head-to-head table with exploitability; submit the strategy from an issued run | Lab 4, 6.7205.4x |

The lab follows the courses' common pattern: changed readiness cases test whether the learner's functions
transfer beyond the notebook's examples; later checkpoints ask what a record must hold, which claims the evidence
supports, and what stays fixed in a next experiment. Exploitability is reported, never required.

## Lab runtime notes

- **Lab 4** runs on CPU threads; a GPU adds nothing. Its game, phantom tic-tac-toe with both opening moves fixed,
  has 76,259 information sets. On free Colab one traversal with the learner's own functions takes about a
  second, and the notebook's computation about 15 minutes in all.
- The submission is the average strategy from the learner's issued run, in the format of the original course
  challenge: a zipped NumPy array, one row per information set, nine columns.

## The game package

Lab 4 uses [exp-a-spiel](https://github.com/gabrfarina/exp-a-spiel) by Gabriele Farina (MIT license), with a patch
that adds phantom tic-tac-toe with a fixed opening. [`third_party/exp-a-spiel/`](third_party/exp-a-spiel/) holds
the patch, the upstream license and the commit it applies to. This repository's CI builds a wheel from upstream
plus the patch and attaches it to a release; the notebook installs that wheel and checks its digest.

## For maintainers

Do not edit the notebooks or `labs/helpers/` by hand. `Optimization/_workfiles/practice/build_practice.py` in the
private course repository writes each notebook with its solved reference copy, which stays private, and
`make_lab4_helpers.py` copies the provided code here. Nothing with a solution in it belongs here. A changed patch
needs a new release tag; never move a tag a published notebook reads.
