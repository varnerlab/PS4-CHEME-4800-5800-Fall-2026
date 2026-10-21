# Student implementation | Complete the four functions in Parts 1 and 2.
# Already written for you: Data.jl loads the images; Trainers.jl holds the training
# functions my_perceptron and my_logistic_regression from the L9b and L9d labs;
# Reporting.jl writes the plots and tables.

# Do not change the function names, arguments, or return types.
# Replace each placeholder error(...) with your implementation and return value.
# X is the augmented data matrix: one row per image, the 784 pixel values, then a
# 1 for the bias.
# labels holds the true digit, 0 to 9, of each image.
# W holds one model per digit: column d + 1 is the model for digit d,
# because Julia counts from 1.

# ----------------------------------------------------------------------------
# Part 1a: Labels y = ±1 for the model of one digit
# ----------------------------------------------------------------------------

"""
    digit_labels(labels::Array{Int64,1}, d::Int64) -> Array{Float64,1}

Turn the digit labels into the labels `y = ±1` for the model of digit `d`: `y = 1`
for an image of `d` and `y = -1` for an image of any other digit.

### Arguments
- `labels::Array{Int64,1}`: the true digit of each image, each an integer from 0 to 9.
- `d::Int64`: the digit whose images get `y = 1`, an integer from 0 to 9.

### Returns
A new `Array{Float64,1}` with one entry per label: `1.0` where the label is `d` and
`-1.0` everywhere else. The training functions need labels in this form. An empty
`labels` gives an empty vector. Leave `labels` unchanged.

### Errors
Throws an `ArgumentError` if `d` is outside 0 to 9 or if any label is outside 0 to 9.

### Example
```julia
y = digit_labels([3, 1, 3, 0], 3); # [1.0, -1.0, 1.0, -1.0]
```
"""
function digit_labels(labels::Array{Int64,1}, d::Int64)

    # TODO 1: Build and return the labels y = ±1.

    # Check the inputs -
    # Throw an ArgumentError for a d or a label outside 0 to 9.

    # Build the labels -
    # Use 1.0 where the label equals d and -1.0 everywhere else.

    error("Complete digit_labels in src/Compute.jl");
end


# ----------------------------------------------------------------------------
# Part 1b: Train one model per digit
# ----------------------------------------------------------------------------

"""
    train_digit_models(X::Array{Float64,2}, labels::Array{Int64,1};
        method::Symbol = :logistic, maxiter::Int64) -> Array{Float64,2}

Train ten binary classifiers, one for each digit from 0 to 9.

### Arguments
- `X::Array{Float64,2}`: the augmented data matrix, one row per image: the pixel
  values, followed by a 1 for the bias.
- `labels::Array{Int64,1}`: the true digit of each image, in the same order as the
  rows of `X`.
- `method::Symbol = :logistic`: `:logistic` for `my_logistic_regression`, or
  `:perceptron` for `my_perceptron`.
- `maxiter::Int64`: passed to the training function. For logistic regression it is
  the largest number of gradient-descent steps; for the Perceptron it is the largest
  number of passes through the images. Either function can stop early. Required.

### Returns
A new `Array{Float64,2}` `W` with `size(X, 2)` rows and 10 columns. Column `d + 1`
holds the parameters `θ` of the model for digit `d` (the pixel weights, followed by
the bias): column 1 is the model for 0 and column 10 is the model for 9. Leave `X`
and `labels` unchanged.

### Method
For each digit `d` from 0 to 9, make the labels `y = digit_labels(labels, d)`
and train one model with the selected training function:

- logistic regression: `my_logistic_regression(X, y; α = α, maxiter = maxiter)`,
  where `α = 1/opnorm(X)^2` is the L9d learning rate. Compute `α` once, before the
  loop over digits, because `opnorm` takes a moment on 10,000 images.
- Perceptron: `my_perceptron(X, y; maxiter = maxiter)`.

Store the returned `θ` in column `d + 1` of `W`. Use these calls exactly, so every
correct solution gives the same parameters. A digit with no images still gets a
column, trained on labels that are all `-1.0`.

### Errors
Throws an `ArgumentError` for a `method` other than `:logistic` or `:perceptron`, and
a `DimensionMismatch` if `labels` does not have one entry per row of `X`. Check both
before training. The training functions and `digit_labels` report any other bad input.

### Example
```julia
data = load_digits();
W = train_digit_models(data.training.X, data.training.labels; method = :logistic, maxiter = 1000);
size(W) # (785, 10)
```
"""
function train_digit_models(X::Array{Float64,2}, labels::Array{Int64,1};
    method::Symbol = :logistic, maxiter::Int64)

    # TODO 2: Train the ten models and return W.

    # Check the inputs -
    # Check the method and the number of labels before any training.

    # Train one model per digit -
    # Make W with size(X, 2) rows and 10 columns; do not write 785, because the
    # tests also use tiny images with only a few pixels. For each d from 0 to 9,
    # make the labels y = ±1 with digit_labels, call the training function shown
    # in the Method section, and store its θ in column d + 1. For logistic
    # regression, compute α = 1/opnorm(X)^2 once, before the loop.

    # Return the parameter matrix W -

    error("Complete train_digit_models in src/Compute.jl");
end


# ----------------------------------------------------------------------------
# Part 2a: Read each image as a digit
# ----------------------------------------------------------------------------

"""
    predict_digits(W::Array{Float64,2}, X::Array{Float64,2}) -> Array{Int64,1}

Read each image as the digit whose model gives it the highest score.

### Arguments
- `W::Array{Float64,2}`: the parameters of the ten models, one model per column, as
  returned by `train_digit_models`. Column `d + 1` is the model for digit `d`.
- `X::Array{Float64,2}`: the augmented data matrix, one row per image: the pixel
  values, followed by a 1 for the bias.

### Returns
A new `Array{Int64,1}` with one predicted digit, from 0 to 9, per row of `X`. The
score of image `i` for digit `d` is row `i` of `X` times column `d + 1` of `W`; all
the scores at once are the matrix `X * W`, with one row per image and one column per
digit. Each image is read as the digit with the highest score. If two digits tie,
keep the smaller digit; Julia's `argmax` does this. Zero images give an empty
vector. Leave `W` and `X` unchanged.

For logistic regression, the highest score also gives the highest probability that
the image shows that digit, `σ(2β x̂ᵀθ)` with `β = 1`, because the logistic function
`σ` is increasing.

### Errors
Throws a `DimensionMismatch` unless `W` has `size(X, 2)` rows and 10 columns.

### Example
```julia
ŷ = predict_digits(W, data.testing.X); # one predicted digit per testing image
count(ŷ .== data.testing.labels) # the number of testing images read correctly
```
"""
function predict_digits(W::Array{Float64,2}, X::Array{Float64,2})

    # TODO 3: Return the digit with the highest score for each image.

    # Check the sizes -
    # W needs one row per column of X and one column per digit.

    # Score every image for every digit -
    # X * W has one row per image and one column per digit.

    # Pick the digit with the highest score in each row -
    # The highest score in column d + 1 means digit d. For a matrix S, S[i, :] is
    # row i, and argmax(v) gives the position of the largest entry of a vector v
    # (the first one, on a tie).

    error("Complete predict_digits in src/Compute.jl");
end


# ----------------------------------------------------------------------------
# Part 2b: Count the mix-ups
# ----------------------------------------------------------------------------

"""
    confusion_matrix(actual::Array{Int64,1}, predicted::Array{Int64,1}) -> Array{Int64,2}

Count how often each true digit was read as each digit.

### Arguments
- `actual::Array{Int64,1}`: the true digit of each image, from 0 to 9.
- `predicted::Array{Int64,1}`: the digit the model read for each image, in the same
  order.

### Returns
A new 10 × 10 `Array{Int64,2}`, the confusion matrix. As in the labs, rows are the
true digits and columns are the predicted digits. Entry `[a + 1, p + 1]` counts the
images whose true digit is `a` and that the model read as `p`. The diagonal counts
the correct readings, and each entry off the diagonal counts one kind of mix-up.
Empty inputs give a matrix of zeros. Leave `actual` and `predicted` unchanged.

### Errors
Throws a `DimensionMismatch` if `actual` and `predicted` have different lengths, and
an `ArgumentError` if any value in either one is outside 0 to 9.

### Example
```julia
CM = confusion_matrix(data.testing.labels, ŷ); # important: actual digits first, predicted digits second
CM[5, 10] # the number of 4s read as 9
```
"""
function confusion_matrix(actual::Array{Int64,1}, predicted::Array{Int64,1})

    # TODO 4: Build and return the 10 × 10 confusion matrix.

    # Check the inputs -
    # Check the lengths, then check that no value is outside 0 to 9.

    # Count each pair (true digit, digit read) -
    # Start from zeros(Int, 10, 10), a 10 × 10 matrix of whole-number zeros;
    # zeros(10, 10) would hold decimals, but the result must be an Array{Int64,2}.
    # For each image, add one to row a + 1 and column p + 1, where a is its true
    # digit and p is the digit read.

    # Return the confusion matrix -

    error("Complete confusion_matrix in src/Compute.jl");
end
