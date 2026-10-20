# Problem Set 4 (PS4): Reading a plant logbook

A chemical plant kept handwritten logbooks for 30 years. Every hour, an operator
wrote down the reactor temperature, the pressure, and the flow rates. The plant
now wants those records in a spreadsheet so its engineers can study them. Nobody
wants to type 30 years of numbers by hand, so you will teach a computer to read
the handwriting, one digit at a time.

In Week 9, we explored two binary classifiers, models that answer a yes/no
question: the Perceptron and logistic regression.
A logbook digit has ten possible values, `0` through `9`. You will build a digit reader out of ten yes/no
models, one for each digit, and then check how often it reads the logbook correctly.
Thus, we will solve a multi-class classification problem using a collection of binary classifiers.

Problem Set 4 (PS4) is due: __11:59 PM ET, Saturday, November 7, 2026__. Released October 24, 2026.

## Setup

Use Julia `1.12.7`. Open this folder in VS Code, open a terminal, and run these
two commands:

```bash
julia --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate()'
julia --startup-file=no --project=. check_submission.jl
```

The first command installs the packages and takes several minutes. The second
runs the 48 tests. They will fail until you finish Parts 1 and 2.

You write your code in [src/Compute.jl](src/Compute.jl) and your answers in
[responses.md](responses.md). Do not edit any other file, with one exception: if
you put helper functions in a new file in `src/`, add that file to
[Include.jl](Include.jl), which shows an example.

We wrote the Perceptron and logistic regression training functions for you:
`my_perceptron` and `my_logistic_regression` in [src/Trainers.jl](src/Trainers.jl).
Do not change that file. We grade with our own copy of it.

## The data

The images come from MNIST, a free public collection of handwritten digits. Each
image is 28 × 28 pixels (see [Source](#source)). Run `data = load_digits()`
([src/Data.jl](src/Data.jl)) to get two sets of images: a training set to train
the models and a testing set to check them.

| Field | Meaning |
|---|---|
| `data.training.X` | The 10,000 training images: one image per row, 785 columns. |
| `data.training.labels` | The true digit (0 to 9) in each training image. |
| `data.testing.X`, `data.testing.labels` | 10,000 different images and their digits. They are used only to check the reader, never to train it. |

`X` is the **augmented data matrix**, as in the Week 9 labs. Each row has 785
numbers: the image's 784 pixel values, from 0 (white paper) to 1 (black ink),
and then a 1. That last 1 lets each model add a constant, called the **bias**.

A yes/no model is a vector $\boldsymbol{\theta}$ of 785 **parameters**: one
**weight** for each pixel, then the bias. The model gives each image $\hat{\mathbf{x}}$ (one
row of `X`) a **score**, $s = \hat{\mathbf{x}}^{\top}\boldsymbol{\theta}$. A
positive score means "yes" and a negative score means "no". The farther the score
is from zero, the more sure the model is. A pixel's weight sets how much ink at
that pixel raises the score (a positive weight) or lowers it (a negative weight).

You will store the ten models side by side as the columns of a 785 × 10 matrix
`W`. **Column `d + 1` holds the model for digit `d`**, because Julia counts from
1: column 1 is the model for 0, and column 10 is the model for 9. In your code,
write `size(X, 2)` instead of 785, because the tests also use tiny images with
only a few pixels.

## Part 1: From two answers to ten

A yes/no model answers one question, such as "Is this a 3?". To train it, label
every image of a 3 with `1` and all the other images with `-1`. Do this for each
of the ten digits and you have ten models. In Part 2, you score each image with
all ten models and keep the digit with the highest score.

Implement in [src/Compute.jl](src/Compute.jl):

1. **`digit_labels(labels, d)`** returns a vector with `1.0` where the label is `d`
   and `-1.0` everywhere else.
2. **`train_digit_models(X, labels; method=:logistic, maxiter)`** trains one model
   for each digit, from 0 to 9, and returns `W`. Set `method` to `:logistic` for
   logistic regression or `:perceptron` for the Perceptron. Make each digit's
   labels with `digit_labels`, and train each model with the exact lines below.
   That way, every correct solution gets the same parameters:

```julia
# Inside src/Compute.jl, opnorm is ready to use. Elsewhere, run `using LinearAlgebra` first.
α = 1/opnorm(X)^2; # the L9d learning rate; compute it once, before the loop over digits
y = digit_labels(labels, d); # inside the loop over d = 0, 1, ..., 9
θ = my_logistic_regression(X, y; α = α, maxiter = maxiter).θ; # method = :logistic
θ = my_perceptron(X, y; maxiter = maxiter).θ;                  # method = :perceptron
```

`maxiter` limits how long each model trains. For logistic regression, it is the
largest number of gradient-descent steps. For the Perceptron, it is the largest
number of passes through all the images. Either function can stop early, as in
the labs: logistic regression when a step barely changes $\boldsymbol{\theta}$,
and the Perceptron after a pass with no mistakes.

The description above each function in [src/Compute.jl](src/Compute.jl) lists
the bad inputs it must reject and the error to throw. Test with:

```bash
julia --startup-file=no --project=. testme_part_1.jl
```

## Part 2: Reading digits and finding mistakes

To read an image, compute its score for all ten digits and keep the digit with
the highest score. The matrix `X * W` holds every score at once: one row per
image and one column per digit. For logistic regression, the highest score also
gives the highest probability, because a larger score always means a larger
probability.

A **confusion matrix** counts the reader's mistakes by type, as in the Week 9
labs. It has one row for each true digit and one column for each digit the model read. The entry in row
`a + 1` and column `p + 1` counts the images of digit `a` that the model read as
`p`. The entries on the diagonal count the correct readings. Every other entry
counts one kind of mistake, such as a 4 read as a 9.

Implement in [src/Compute.jl](src/Compute.jl):

3. **`predict_digits(W, X)`** returns the digit (0 to 9) with the highest score
   for each row of `X`. If two digits tie, keep the smaller one; Julia's `argmax`
   does this.
4. **`confusion_matrix(actual, predicted)`** returns the 10 × 10 confusion matrix
   as an `Array{Int64,2}`.

The description above each function lists the rest of the rules. When both
functions are done, run:

```bash
julia --startup-file=no --project=. testme_part_2.jl
julia --startup-file=no --project=. runlogbook.jl
```

`runlogbook.jl` uses your code to build two digit readers from the 10,000
training images: one with logistic regression (at most 1000 steps per digit) and
one with the Perceptron (at most 20 passes per digit). It then reads the 10,000
testing images with both readers and writes the files below to `outputs/`. It also
prints each reader's **accuracy**, the fraction of testing digits it reads
correctly, and a text copy of the logbook page. It takes about a minute.

| File | Contents |
|---|---|
| `accuracy.csv` | Each reader's accuracy. Rows with digit `all` give the accuracy on all testing digits; the other rows give it for one digit at a time. |
| `confusion-matrix.csv`, `.png` | The confusion matrix for logistic regression. |
| `misread-digits.png` | Logistic regression's six most common mistakes, such as a 5 read as a 3, with four example images of each. Each image shows its true digit and the digit the model read. |
| `digit-weights.png` | Each digit's logistic regression weights, drawn as a 28 × 28 picture. Blue pixels raise the score; red pixels lower it. |
| `logbook-page.csv`, `.png` | A made-up logbook page: 20 four-digit readings written with testing images, next to what logistic regression read. |
| `logbook-summary.csv` | Two results for logistic regression: the fraction of the 10,000 testing digits it reads correctly, and the fraction of four-digit entries it reads with no mistakes. The entries are the testing digits taken four at a time, in order, which gives 2,500 entries. |

Use these files to answer the three questions in [responses.md](responses.md).

## Check and submit

Run the checker:

```bash
julia --startup-file=no --project=. check_submission.jl
```

The checker runs the 48 tests, checks that you answered all three questions, and
writes `MANIFEST.txt`, a record of your results and files. It also deletes the
old files in `outputs/`. If every test passes, it makes new ones the same way
`runlogbook.jl` does, which adds about a minute. The checker does not submit
anything for you.

Zip the whole folder, including `MANIFEST.txt` and `outputs/`. Name the file
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
