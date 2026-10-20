# Problem Set 4 (PS4): Reading a plant logbook

A chemical plant kept handwritten logbooks for 30 years. Every hour, an operator
wrote down the reactor temperature, the pressure, and the flow rates. The plant
now wants those records in a spreadsheet so its engineers can study them. Nobody
wants to type 30 years of numbers by hand, so you will teach a computer to read
the handwriting, one digit at a time.

In the Week 9 labs, you trained two classifiers that answer a yes/no question:
the Perceptron (L9b) and logistic regression (L9d). A logbook digit has ten
possible values, 0 through 9. You will build a digit reader out of ten yes/no
models, one for each digit, and then check how often it reads the logbook
correctly.

**Due: 11:59 PM ET, Saturday, November 7, 2026.** Released October 24, 2026.

## Setup

Use Julia `1.12.7`. Open this folder in VS Code and run these commands from it:

```bash
julia --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate()'
julia --startup-file=no --project=. check_submission.jl
```

The first command installs the packages and takes several minutes. The second
runs the 48 tests, which fail until you finish Parts 1 and 2.

Edit only [src/Compute.jl](src/Compute.jl), [responses.md](responses.md), and, if
you add helper files in `src/`, [Include.jl](Include.jl). The two trainers are
supplied in [src/Trainers.jl](src/Trainers.jl). Do not change them; grading uses
the supplied copies.

## The data

The images come from MNIST, a public set of 28 × 28 pictures of handwritten
digits (see [Source](#source)). Calling `data = load_digits()`
([src/Data.jl](src/Data.jl)) returns a training set and a test set:

| Field | Meaning |
|---|---|
| `data.training.X` | 10,000 training images, one per row, and 785 columns. |
| `data.training.labels` | The true digit (0 to 9) in each training image. |
| `data.test.X`, `data.test.labels` | 10,000 test images and their digits, used only to check the reader. |

Each row of `X` holds one image's 784 pixel values, from 0 (white paper) to 1
(black ink), followed by a 1 for the bias, as in the L9 labs.

A yes/no model is a vector $\boldsymbol{\theta}$ of 785 numbers: one **weight** per
pixel, then the bias. A pixel's weight says how much ink at that pixel raises or
lowers the score. The model's **score** for an image $\hat{\mathbf{x}}$ (one
row of `X`) is $s = \hat{\mathbf{x}}^{\top}\boldsymbol{\theta}$. A positive score
means "yes" and a negative score means "no"; the farther from zero, the surer the
model.

You will store the ten models as the columns of a 785 × 10 matrix `W`. (The tests
also use tiny images with only a few pixels, so use `size(X, 2)` in your code, not
785.)
**Column `d + 1` holds the model for digit `d`**, because Julia counts from 1:
column 1 is the model for 0, and column 10 is the model for 9.

## Part 1: From two answers to ten

A yes/no model answers one question, such as "Is this a 3?". To train it, label
every image of a 3 with `1` and all the other images with `-1`. Repeat this for
each digit and you have ten models. In Part 2, you score each image with all ten
models and keep the digit with the highest score.

Implement in [src/Compute.jl](src/Compute.jl):

1. **`digit_labels(labels, d)`** returns a vector with `1.0` where the label is `d`
   and `-1.0` everywhere else.
2. **`train_digit_models(X, labels; method=:logistic, maxiter)`** trains one model
   per digit, from 0 to 9, and returns `W`. The `method` is `:logistic` or
   `:perceptron`. Use `digit_labels` for the labels and these calls for the
   trainers, so every correct solution gives the same weights:

```julia
# Outside src/Compute.jl, run `using LinearAlgebra` first to get opnorm.
α = 1/opnorm(X)^2; # the L9d learning rate; compute it once, before the loop over digits
y = digit_labels(labels, d); # inside the loop over d = 0, 1, ..., 9
θ = my_logistic_regression(X, y; α = α, maxiter = maxiter).θ; # method = :logistic
θ = my_perceptron(X, y; maxiter = maxiter).θ;                  # method = :perceptron
```

For logistic regression, `maxiter` is the most gradient-descent steps the trainer
takes. For the Perceptron, it is the most passes through the images. Either
trainer can stop sooner, as in the labs. Each docstring lists the input checks
and errors. Test with:

```bash
julia --startup-file=no --project=. testme_part_1.jl
```

## Part 2: Reading digits and finding mistakes

To read an image, compute its score for all ten digits and keep the digit with
the highest score. The scores of every image for every digit form the matrix
`X * W`: one row per image and one column per digit. For logistic regression, the
highest score is also the highest probability, because a larger score always
gives a larger probability.

A **confusion table** counts the reader's mix-ups. It has one row for each true
digit and one column for each digit the model read. The entry in row `a + 1` and
column `p + 1` counts the images of digit `a` that the model read as `p`. The
diagonal counts the correct readings; each entry off the diagonal counts one kind
of mistake.

Implement in [src/Compute.jl](src/Compute.jl):

3. **`predict_digits(W, X)`** returns the digit (0 to 9) with the highest score
   for each row of `X`. If two digits tie, keep the smaller one; Julia's `argmax`
   does this.
4. **`confusion_table(actual, predicted)`** returns the 10 × 10 confusion table
   as a `Matrix{Int}`.

The docstrings list the remaining requirements. Then run:

```bash
julia --startup-file=no --project=. testme_part_2.jl
julia --startup-file=no --project=. runlogbook.jl
```

`runlogbook.jl` trains both methods on the 10,000 training images (at most 1000
gradient-descent steps or 20 Perceptron passes per digit), reads the 10,000 test
images, and writes these files to `outputs/`. It takes about a minute. It also
prints both accuracies and a text copy of the logbook page.

| File | Contents |
|---|---|
| `accuracy.csv` | The fraction of test digits each method reads correctly: the rows with digit `all` are overall, and the other rows are for each digit. |
| `confusion-table.csv`, `.png` | The confusion table for logistic regression. |
| `misread-digits.png` | Examples of the six most common mix-ups by logistic regression, four of each, labeled with the true digit and the digit read. |
| `digit-weights.png` | Each digit's logistic-regression weights, drawn as a 28 × 28 picture. |
| `logbook-page.csv`, `.png` | A logbook page of 20 four-digit readings written with test digits, and what logistic regression read. |
| `logbook-summary.csv` | The fraction of all 10,000 test digits that logistic regression reads correctly, and the fraction of 2,500 four-digit entries, made by taking the test digits four at a time in order, that it reads completely correctly. |

Use these files to answer the three questions in [responses.md](responses.md).

## Check and submit

Run the checker:

```bash
julia --startup-file=no --project=. check_submission.jl
```

It runs the 48 tests and checks that all three questions are answered. It always
writes `MANIFEST.txt`. It clears old reports from `outputs/` and makes new ones
only when all the tests pass; then it trains both readers, as `runlogbook.jl`
does, so it takes about a minute longer. It runs only on your computer; it does not submit
anything.

Zip the whole folder, including `MANIFEST.txt` and `outputs/`, as
`CHEME-4800-5800-PS4-<netid>.zip` (for example, `CHEME-4800-5800-PS4-abc123.zip`)
and upload it to Canvas. If some tests still fail at the deadline, submit anyway
for partial credit. See [RUBRIC.md](RUBRIC.md) for the grading scale.

## Policies

- **Revisions:** Submit something by the deadline, even if it does not run. You
  may then revise and resubmit until the end of the semester; your highest score
  counts. No initial submission means a score of `0` and no revisions.
- **Reference solution:** Released after the deadline. Use it to understand and
  debug your work; copying it results in a score of `0`.
- **Collaboration and AI:** Discuss ideas with classmates, but do not share code
  or solutions. You may use Julia documentation, AI tools, and internet
  resources. You must be able to explain your code.

## Source

The digit images come from the MNIST database of handwritten digits by Yann
LeCun, Corinna Cortes, and Christopher J. C. Burges
([dataset page](http://yann.lecun.com/exdb/mnist/)), described in **LeCun,
Bottou, Bengio, and Haffner (1998), "Gradient-based learning applied to document
recognition," Proceedings of the IEEE, 86(11), 2278–2324**
([DOI: 10.1109/5.726791](https://doi.org/10.1109/5.726791)). MNIST is shared under
the Creative Commons Attribution-Share Alike 3.0 license; see
[THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt). The plant and its logbook are
made up for this problem set. The [data notes](data/README.md) describe the files
and how the logbook page was built.
