# PS4 reference solution

This directory holds the reference solution for PS4: Reading a plant logbook.

- [src/Compute.jl](src/Compute.jl) completes the four functions:
  `digit_labels` and `train_digit_models` for Part 1, and `predict_digits` and
  `confusion_matrix` for Part 2.
- [responses.md](responses.md) has model answers to the three written questions.

Use the reference solution to understand mistakes and debug your own work, but
do not copy it. Copying the reference solution results in a score of `0` for
this assignment.

## Running the reference solution

The easiest way is to download `PS4-CHEME-4800-5800-Fall-2026-solution.zip`
from the repository's solution release. That ZIP already has this `Compute.jl`
in `src/` and the model answers in `responses.md`; see its README.

To run it in your own PS4 folder instead, copy this `solution` directory into
that folder, next to `src/`. Then run this command from your PS4 folder:

```bash
julia --startup-file=no --project=. check_submission.jl --solution
```

The `--solution` flag loads `solution/src/Compute.jl` and `solution/responses.md`
in place of your files. All 48 tests should pass, and the status should read
`REFERENCE SOLUTION PASSED`. The plots and tables go to `solution/outputs/`;
your own code, responses, and outputs are not changed.

The reference solution gives these results on the 10,000 testing images:

| Measure | Value |
|---|---:|
| Testing accuracy, logistic regression (1000 steps) | 0.8974 |
| Testing accuracy, Perceptron (20 passes) | 0.8683 |
| Four-digit entries read correctly, logistic regression | 0.6672 |
