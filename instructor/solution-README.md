# PS4 reference solution: Reading a plant logbook

This archive contains the completed reference solution for CHEME 4800/5800,
Fall 2026. The solution is already in [src/](src/), so the checker runs both
parts without moving or editing any files.

Use Julia `1.12.7`. From the folder containing this README, run:

```sh
julia --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate()'
julia --startup-file=no --project=. check_submission.jl
```

The first command installs the packages and takes several minutes. The checker
should report that all 48 tests pass and that the status is `READY TO PACKAGE`.
It writes the plots and tables to `outputs/` and a local `MANIFEST.txt`; it
does not upload anything.

The archive includes:

- [src/Compute.jl](src/Compute.jl): the completed, commented reference code for
  `digit_labels`, `train_digit_models`, `predict_digits`, and
  `confusion_matrix`. The other files in `src/` are the supplied code.
- [responses.md](responses.md): model answers to the three written questions.
- [starter/Compute.jl](starter/Compute.jl): the original starter, for comparison.
- [ASSIGNMENT.md](ASSIGNMENT.md): the original problem statement. Its
  directions to complete the code apply to your own submission.
- [data/](data/README.md): the MNIST digit images and the logbook page.
- [RUBRIC.md](RUBRIC.md): grading rules and completion requirements.
- [testme_part_1.jl](testme_part_1.jl) and [testme_part_2.jl](testme_part_2.jl):
  the public tests.

To write only the plots and tables, run
`julia --startup-file=no --project=. runlogbook.jl`.

| Measure | Value |
|---|---:|
| Testing accuracy, logistic regression (1000 steps) | 0.8974 |
| Testing accuracy, Perceptron (20 passes) | 0.8683 |
| Four-digit entries read correctly, logistic regression | 0.6672 |

Use this reference to understand mistakes and debug your own work. You may
study it, but you may not copy it. Copying the reference solution results in a
score of `0` for this assignment.
