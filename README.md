# Problem Set 4 (PS4): Reading a plant logbook

Every hour, an operator wrote down the reactor temperature, the pressure, and the flow rates for a chemical process over a 30-year period. 
The plant manager now wants those records in a spreadsheet so the process engineers can study them, and optimize the process. However, nobody
wants to type 30 years of numbers by hand, so you (as a new intern) will teach a computer to read the handwriting, one digit at a time.

You'll do this by training a collection of binary classifiers on a public dataset of handwritten digits, then using them to read the logbook. 
In Week 9, we explored two binary classifiers, models that answer a yes/no question: the Perceptron and logistic regression.
A logbook digit has ten possible values, `0` through `9`. You will build a digit reader out of ten yes/no
models, one for each digit, and then check how often it reads the logbook correctly.

Each of the ten models is a linear binary classifier for one digit $d$: it
answers the question "Is this image a $d$?". Its labels are $y=1$ for an image of $d$ and
$y=-1$ for an image of any other digit. Thus, we will solve a multi-class classification problem using a collection of binary classifiers.

Problem Set 4 (PS4) is due: __11:59 PM ET, Saturday, November 7, 2026__. Released October 24, 2026.

## Setup

Use Julia `1.12.7`. Open this folder in VS Code, open a terminal, and run:

```bash
julia --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate()'
julia --startup-file=no --project=. check_submission.jl
```

The first command installs the packages (several minutes). The second runs the
48 tests, which fail until you finish Parts 1 and 2.

Write your code in [src/Compute.jl](src/Compute.jl) and your answers in
[responses.md](responses.md). Edit no other file, with one exception: if you put
helper functions in a new file in `src/`, add it to [Include.jl](Include.jl),
which shows an example.

The training functions, `my_perceptron` and `my_logistic_regression`, are already
written in [src/Trainers.jl](src/Trainers.jl). Do not change that file.

## Materials and Methods

The images come from MNIST, a public collection of handwritten digits, 28 × 28
pixels each (see [Source](#source)). Calling `data = load_digits()`
([src/Data.jl](src/Data.jl)) returns a training set and a testing set:

| Field | Meaning |
|---|---|
| `data.training.X` | The 10,000 training images: one image per row, 785 columns. |
| `data.training.labels` | The true digit (0 to 9) in each training image. |
| `data.testing.X`, `data.testing.labels` | 10,000 other images and their digits, used only to check the reader. |

The matrix `X` is the **augmented data matrix**. Each row has 785
numbers: the image's 784 pixel values, from 0 (white paper) to 1 (black ink),
and then a 1. That last 1 lets each model add a **bias** term.

Each model has **parameters** $\mathbf{\theta}=\left(w_{1},\dots,w_{784},b\right)$: one **weight** $w_{j}$ per
pixel, then the bias $b$. For an image $\hat{\mathbf{x}}$ (one row of `X`), it
passes the **score** $\hat{\mathbf{x}}^{\top}\mathbf{\theta}$ through an
activation function $\sigma$:

- **Perceptron (L9a):** $\sigma(z)=\texttt{sign}(z)$, so the estimated label is
  $\hat{y}=\texttt{sign}\left(\hat{\mathbf{x}}^{\top}\mathbf{\theta}\right)=\pm 1$.
- **Logistic regression (L9c):** $\sigma(z)=1/\left(1+e^{-z}\right)$, the logistic
  function, so the probability that the image is a $d$ is
  $P_{\mathbf{\theta}}(y=1\mid\hat{\mathbf{x}})=\sigma\left(2\beta\hat{\mathbf{x}}^{\top}\mathbf{\theta}\right)$,
  with $\beta=1$ in PS4.

We'll store the parameters of the ten models as the columns of a 785 × 10 matrix `W`. **Column `d + 1`
holds the model for digit `d`**, because Julia counts from 1: column 1 is digit
0, and column 10 is digit 9. In your code, write `size(X, 2)`, not 785, because
the tests also use tiny images with only a few pixels.

For both methods, a positive score means the model reads the image as a $d$, and
a negative score means it reads it as some other digit; the farther the score is
from zero, the more confident the model. 

## Part 1: From two answers to ten

To train the model for 3, label every image of a 3 with $y=1$ and every other
image with $y=-1$. Repeat for each digit to get ten models. In Part 2, you score each image with all ten and keep
the digit with the highest score.

Implement in [src/Compute.jl](src/Compute.jl):

1. **`digit_labels(labels, d)`:** returns `1.0` where the label is `d` and `-1.0`
   elsewhere.
2. **`train_digit_models(X, labels; method=:logistic, maxiter)`:** trains one
   model per digit and returns `W`. Use `method = :logistic` for logistic
   regression or `method = :perceptron` for the Perceptron. Build the labels with
   `digit_labels` and train with exactly these lines, so every correct solution
   gets the same parameters:

```julia
# Inside src/Compute.jl, opnorm is ready to use; elsewhere, run `using LinearAlgebra` first.
α = 1/opnorm(X)^2; # the L9d learning rate; compute it once, before the loop over digits
y = digit_labels(labels, d); # inside the loop over d = 0, 1, ..., 9
θ = my_logistic_regression(X, y; α = α, maxiter = maxiter).θ; # method = :logistic
θ = my_perceptron(X, y; maxiter = maxiter).θ;                  # method = :perceptron
```

The keyword `maxiter` caps training: the most gradient-descent steps for logistic
regression, or the most passes through the images for the Perceptron. As in the
labs, either function can stop early: logistic regression when a step barely
changes $\mathbf{\theta}$, and the Perceptron after a pass with no mistakes.

The description above each function in [src/Compute.jl](src/Compute.jl) lists
the inputs it must reject and the error to throw. Test with:

```bash
julia --startup-file=no --project=. testme_part_1.jl
```

## Part 2: Reading digits and finding mistakes

To read an image, score it with all ten models and keep the digit with the
highest score. The matrix `X * W` holds every score: one row per image, one
column per digit. For logistic regression, the highest score also gives the
highest probability $P_{\mathbf{\theta}}(y=1\mid\hat{\mathbf{x}})$, because
$\sigma$ is increasing.

A **confusion matrix**, as in the Week 9 labs, counts the reader's mistakes by
type. Its rows are the true digits and its columns are the digits read: the entry
in row `a + 1` and column `p + 1` counts the images of digit `a` read as `p`. The
diagonal counts the correct readings; every other entry counts one kind of
mistake, such as a 4 read as a 9.

Implement in [src/Compute.jl](src/Compute.jl):

3. **`predict_digits(W, X)`:** returns the digit (0 to 9) with the highest score
   for each row of `X`. On a tie, keep the smaller digit, as Julia's `argmax`
   does.
4. **`confusion_matrix(actual, predicted)`:** returns the 10 × 10 confusion
   matrix as an `Array{Int64,2}`.

The description above each function lists the remaining rules. When both
functions are done, run:

```bash
julia --startup-file=no --project=. testme_part_2.jl
julia --startup-file=no --project=. runlogbook.jl
```

The script `runlogbook.jl` trains two digit readers on the 10,000 training
images: logistic regression (at most 1000 steps per digit) and the Perceptron (at
most 20 passes per digit). It reads the 10,000 testing images with both, prints
each reader's **accuracy** (the fraction of testing digits read correctly) and a
text copy of the logbook page, and writes these files to `outputs/` in about a
minute:

| File | Contents |
|---|---|
| `accuracy.csv` | Each reader's accuracy, overall (rows with digit `all`) and for each digit. |
| `confusion-matrix.csv`, `.png` | The confusion matrix for logistic regression. |
| `misread-digits.png` | Four examples of each of logistic regression's six most common mistakes, one direction per row (such as 5 read as 3). |
| `digit-weights.png` | Each digit's logistic regression weights as a 28 × 28 picture. Blue pixels raise the score; red pixels lower it. |
| `logbook-page.csv`, `.png` | A made-up logbook page of 20 four-digit readings written with testing images, and what logistic regression read. |
| `logbook-summary.csv` | For logistic regression, the fraction of the 10,000 testing digits read correctly and the fraction of four-digit entries read with no mistakes. The 2,500 entries are the testing digits taken four at a time, in order. |

Use these files to answer the three questions in [responses.md](responses.md).

## Check and submit

Run the checker:

```bash
julia --startup-file=no --project=. check_submission.jl
```

The checker runs the 48 tests, checks that you answered all three questions, and
writes `MANIFEST.txt`, a record of your results and files. It also deletes the
old reports in `outputs/` and, if every test passes, makes new ones as
`runlogbook.jl` does (about a minute more). It does not submit anything.

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
([DOI: 10.1109/5.726791](https://doi.org/10.1109/5.726791)). 

MNIST is shared under the Creative Commons Attribution-Share Alike 3.0 license; see
[THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt). The plant and its logbook are
made up for this problem set. The [data notes](data/README.md) describe the files
and how the logbook page was built.
