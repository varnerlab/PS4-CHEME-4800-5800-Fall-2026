# PS4 instructor notes

The student task is fixed: build a ten-digit reader for handwritten MNIST digits
out of ten yes/no models, compare logistic regression with the Perceptron, and
see how digit errors spoil whole logbook entries. Parts 1–2 require four
functions (`digit_labels`, `train_digit_models`, `predict_digits`,
`confusion_matrix`) and three written responses, graded out of four points using
48 tests. There is no Advanced track. The two trainers are supplied: they are the
L9b Perceptron and the L9d gradient-descent logistic regression, with the lab
names, arguments, and results. The design is recorded in [DESIGN.md](DESIGN.md)
and the build plan in [plans/](plans/); both are local files, ignored by Git,
because DESIGN.md lists the ideas expected in the answer to question 2.

## Reviewing the assignment

- Read [the problem statement](../README.md), [rubric](../RUBRIC.md), and
  [questions](../responses.md).
- Inspect the reference reports in [reference/](reference/):
  [confusion matrix](reference/confusion-matrix.png),
  [logbook page](reference/logbook-page.png),
  [misread digits](reference/misread-digits.png),
  [weight pictures](reference/digit-weights.png), and the CSV tables.
- The [reference solution](../solution/README.md) contains completed code and
  model written responses.
- [The student ZIP](artifacts/PS4-CHEME-4800-5800-Fall-2026-student.zip)
  contains starter code and data, with no instructor files, outputs, or solutions.

Release is Saturday, October 24, 2026. The initial submission deadline is
11:59 PM ET on Saturday, November 7, 2026. No commit, push, tag, GitHub release,
or Canvas upload is performed by these scripts.

Students download the **Source code (zip)** of the tagged GitHub release
`ps4-cheme-4800-5800-2026.1`; `.gitattributes` keeps instructor material out of
that archive. Pushing the tag runs the
[release Action](../.github/workflows/release-ps4.yml), which runs
[validate_release.py](validate_release.py) on the source archive and creates a
**draft** release with the [release notes](release-notes.md). That validator
needs no reference solution; run it locally with
`python3 instructor/validate_release.py` before tagging (it needs a committed
Git tree). Review and publish the draft, then post the
[Canvas description](canvas-assignment-description.html), which links to the
same tag. Later corrections use a new tag.

## Reproducing the checks

To run the reference solution directly from the assignment root:

```bash
julia --startup-file=no --project=. check_submission.jl --solution
```

This runs all 48 tests, checks `solution/responses.md`, and writes the reports to
`solution/outputs/` and a manifest to `solution/MANIFEST.txt`. The student files
and outputs stay unchanged. The same flag works with `testme_part_1.jl`,
`testme_part_2.jl`, and `runlogbook.jl`. The default remains the student
implementation; a missing local solution causes an error rather than falling back
to student code.

From the assignment root, install the Julia dependencies and run:

```bash
julia --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate()'
python3 instructor/validate.py
python3 instructor/build_student_package.py
```

The validator first runs [test_trainers.jl](test_trainers.jl), separate from
student scores. It checks the loader (sizes, pixel range, labels, image
orientation, and a clear error for a cut-short data file) and checks that the
supplied trainers match the L9b and L9d lab reference solutions, which it reads
from the sibling course repository (the comparison is skipped with a warning when
that repository is absent, as in the release Action). The validator then copies
the student files into temporary directories and substitutes the reference
implementation only in those copies. It checks all 48 tests, the driver, and the
checker's handling of unfinished code, partial work, syntax errors, missing
responses, Windows line endings, runs from another folder, and reporting errors.
Logs are in `instructor/validation-output/`; reference reports are copied to
`instructor/reference/`. Both directories are ignored by Git and excluded from the
student package.

To rebuild the data from the original MNIST files, run
`python3 -I instructor/import_data.py`. It downloads the four gzip files from the
mirror, checks their SHA-256 hashes, and rewrites `data/`. It never reads a local
copy: this machine also has Fashion-MNIST files with the same names.

## Reference results

With the reference solution (at most 1000 gradient-descent steps and 20
Perceptron passes per digit, trained on the first 10,000 MNIST training images,
tested on all 10,000 test images):

| Measure | Value |
|---|---|
| Logistic regression test accuracy | 0.8974 (8,974 of 10,000) |
| Perceptron test accuracy | 0.8683 (8,683 of 10,000) |
| Most-confused pairs, both ways (logistic) | 4 and 9: 99 (51 + 48); 3 and 5: 76 (22 + 54); 7 and 9: 71 |
| Four-digit entries read correctly (logistic) | 0.6672 (1,668 of 2,500); 0.8974⁴ = 0.6486 |
| Logbook page | 3 of 20 entries misread: 342.5 → 342.6 K, 1046 → 1066 kPa, 351.2 → 253.7 K |

Logistic-regression test accuracy at other step counts: 0.834 after 50 steps,
0.887 after 500, 0.897 after 1000, and 0.904 after 2000.

Whole-entry accuracy (0.6672) is a little higher than (digit accuracy)⁴
(0.6486), which suggests the mistakes cluster: some four-digit groups hold two
or more of them.

## Releasing the reference solution

The reference solution stayed untracked until the initial submission deadline
passed. It is now tracked in `solution/`; `.gitattributes` still keeps it out of
the student source archive. Before the release, the test scripts, test support,
and checker were rewritten without closures, one-line `||` and `&&` branches, or
ternaries. The old and new test scripts gave identical per-case results on the
starter, the reference, and 18 faulty implementations, and the reference tables
are byte-identical.

The [Solution release Action](../.github/workflows/solution-release.yml) runs
[build_solution_release.py](build_solution_release.py), which builds
`PS4-CHEME-4800-5800-Fall-2026-solution.zip`, extracts it, installs its Julia
environment, and runs both test scripts and the submission checker on its
contents. The ZIP puts the reference `Compute.jl` in `src/`, the model answers in
`responses.md`, and the original starter in `starter/`. Its README comes from
[solution-README.md](solution-README.md). The Action attaches the ZIP and its
SHA-256 checksum to a new **draft** release whose notes come from
[solution-release-notes.md](solution-release-notes.md).

The Action runs when changes to `solution/`, the workflow, the builder, or the
solution release text are pushed to `main`. Its tag is
`ps4-cheme-4800-5800-2026-solutions.N`, where `N` is the workflow run number;
**Actions → Solution release → Run workflow** accepts an explicit new tag. An
existing tag is never overwritten. Review the draft linked in the run summary and
publish it. To build the same ZIP locally, run
`python3 instructor/build_solution_release.py`; the tested ZIP and checksum are
written to the ignored `dist/` directory.

## Modeling choices

- **MNIST, not the 8 × 8 UCI digits.** MNIST looks like real handwriting. A
  design check found that both work well; MNIST makes better pictures.
- **Fixed subset, no randomness.** The first 10,000 training images (863 to 1,127
  per digit) and all 10,000 test images. Every correct submission produces the
  same numbers.
- **The exact trainer calls are in the README**, including `α = 1/opnorm(X)^2`
  (the L9d learning rate with β = 1, δ = 0). The tests compare each column of `W` with
  a direct trainer call to `rtol = 1e-8`. `maxiter` is a required keyword because
  it means steps for logistic regression and passes for the Perceptron, so no
  single default fits both.
- **The supplied trainers keep the lab math.** `my_logistic_regression` computes
  the gradient for all images at once (`X' * ((1 .- σ.(u)) .* y)`) instead of the
  lab's loop over images, and uses an overflow-free form of the same loss.
  `my_perceptron` reads rows from a transposed copy of `X`. `test_trainers.jl`
  confirms the results match the labs (Perceptron exactly; logistic regression to
  `rtol = 1e-10`).
- **1000 steps.** Plain gradient descent with this learning rate is slow to
  settle on raw pixels. 1000 steps keeps logistic regression clearly ahead of the
  Perceptron while the checker stays fast; the reference results above list the
  accuracy at other step counts.
- **The weight pictures are an extra.** With 1000 unregularized steps they are
  blotchy rather than clean digit shapes; the 0 (a red center) and 1 (a blue
  stroke) are clear. No question asks about them.
- **Whole-entry accuracy** groups the 10,000 test images in order, four at a
  time. The logbook page uses the first unused test image of each needed
  digit; its values and image numbers are in `data/logbook.csv`.
- **License.** The MNIST dataset page has no license statement. The widely used
  statement in the Keras MNIST loader documentation says Yann LeCun and Corinna
  Cortes hold the copyright and the data are under CC BY-SA 3.0. That statement is
  recorded in `THIRD_PARTY_NOTICES.txt`, `data/README.md`, and
  `data/provenance.json`. The instructor approved this wording on October 20, 2026.

## Preparing a release

Review the dates in the main README and the student workload, then run the
validation and build scripts. The builder refuses a release package with dates
still marked to be announced. It uses an explicit allowlist and checks that the
four starter errors remain in `src/Compute.jl`.

The reference code, model answers, and generated reference outputs are kept
outside the student archive. Keep the MNIST attribution in
[THIRD_PARTY_NOTICES.txt](../THIRD_PARTY_NOTICES.txt).
