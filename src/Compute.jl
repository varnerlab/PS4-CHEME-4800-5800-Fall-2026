# Student implementation | Complete the four functions in Parts 1 and 2.
# Supplied support: Data.jl loads the images; Trainers.jl holds my_perceptron and
# my_logistic_regression from the L9b and L9d labs; Reporting.jl writes the plots
# and tables.

# Keep the supplied arguments and return types.
# Replace each placeholder error(...) with your implementation and return value.
# X has one row per image: 784 pixel values, then a 1 for the bias.
# labels holds the true digit, 0 to 9, of each image.
# W holds one model per digit: column d + 1 is the model for digit d,
# because Julia counts from 1.

# ----------------------------------------------------------------------------
# Part 1a: Yes/no labels for one digit
# ----------------------------------------------------------------------------

"""
    digit_labels(labels::AbstractVector{<:Integer}, d::Integer) -> Vector{Float64}

Turn the digit labels into yes/no labels for one digit `d`.

# Arguments

- `labels`: the true digit of each image, each an integer from 0 to 9.
- `d`: the digit to label as "yes", an integer from 0 to 9.

# Returns

A new `Vector{Float64}` with one entry per label: `1.0` where the label is `d` and
`-1.0` everywhere else. The supplied trainers use these labels. An empty `labels`
gives an empty vector. Leave `labels` unchanged.

# Errors

Throw `ArgumentError` if `d` is outside 0 to 9 or if any label is outside 0 to 9.
"""
function digit_labels(labels::AbstractVector{<:Integer}, d::Integer)

    # TODO 1: Build and return the yes/no labels.

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
    train_digit_models(X::Matrix{Float64}, labels::AbstractVector{<:Integer};
        method::Symbol=:logistic, maxiter::Integer) -> Matrix{Float64}

Train ten yes/no models, one for each digit from 0 to 9.

# Arguments

- `X`: one row per image: the pixel values, then a 1 for the bias.
- `labels`: the true digit of each image, in the same order as the rows of `X`.
- `method`: `:logistic` (the default) for `my_logistic_regression`, or
  `:perceptron` for `my_perceptron`.
- `maxiter`: passed to the trainer. For logistic regression it is the most
  gradient-descent steps; for the Perceptron it is the most passes through the
  images. Either trainer can stop sooner. Required.

# Returns

A new `Matrix{Float64}` `W` with `size(X, 2)` rows and 10 columns. Column `d + 1`
holds the weights `θ` of the model for digit `d`: column 1 is the model for 0 and
column 10 is the model for 9. Leave `X` and `labels` unchanged.

# Method

For each digit `d` from 0 to 9, make the yes/no labels `y = digit_labels(labels, d)`
and train one model with the selected trainer:

- logistic regression: `my_logistic_regression(X, y; α = α, maxiter = maxiter)`,
  where `α = 1/opnorm(X)^2` is the L9d learning rate. Compute `α` once, before the
  loop over digits, because `opnorm` takes a moment on 10,000 images.
- Perceptron: `my_perceptron(X, y; maxiter = maxiter)`.

Store the trainer's `θ` in column `d + 1` of `W`. Use these calls exactly, so every
correct solution gives the same weights. A digit with no images still gets a
column, trained on labels that are all `-1.0`.

# Errors

Throw `ArgumentError` for a `method` other than `:logistic` or `:perceptron`, and
`DimensionMismatch` if `labels` does not have one entry per row of `X`. Check both
before training. `digit_labels` and the trainers report any other bad input.
"""
function train_digit_models(X::Matrix{Float64}, labels::AbstractVector{<:Integer};
    method::Symbol=:logistic, maxiter::Integer)

    # TODO 2: Train the ten models and return W.

    # Check the inputs -
    # Check the method and the number of labels before any training.

    # Train one model per digit -
    # Make W with size(X, 2) rows and 10 columns; do not write 785, because the
    # tests also use tiny images with only a few pixels. For each d from 0 to 9,
    # make the yes/no labels with digit_labels, call the trainer shown in the
    # Method section, and store its θ in column d + 1. For logistic regression, compute
    # α = 1/opnorm(X)^2 once, before the loop.

    # Return the weights -

    error("Complete train_digit_models in src/Compute.jl");
end


# ----------------------------------------------------------------------------
# Part 2a: Read each image as a digit
# ----------------------------------------------------------------------------

"""
    predict_digits(W::AbstractMatrix{<:Real}, X::AbstractMatrix{<:Real}) -> Vector{Int}

Read each image as the digit whose model gives it the highest score.

# Arguments

- `W`: the ten models, one per column, as returned by `train_digit_models`.
  Column `d + 1` is the model for digit `d`.
- `X`: one row per image: the pixel values, then a 1 for the bias.

# Returns

A new `Vector{Int}` with one digit from 0 to 9 per row of `X`. The score of image
`i` for digit `d` is row `i` of `X` times column `d + 1` of `W`; all the scores at
once are the matrix `X * W`, with one row per image and one column per digit.
Each image is read as the digit with the highest score. If two digits tie, keep
the smaller digit; Julia's `argmax` does this. Zero images give an empty vector.
Leave `W` and `X` unchanged.

For logistic regression, the highest score is also the highest probability,
because a larger score always gives a larger probability.

# Errors

Throw `DimensionMismatch` unless `W` has `size(X, 2)` rows and 10 columns.
"""
function predict_digits(W::AbstractMatrix{<:Real}, X::AbstractMatrix{<:Real})

    # TODO 3: Return the digit with the highest score for each image.

    # Check the sizes -
    # W needs one row per column of X and one column per digit.

    # Score every image for every digit -
    # X * W has one row per image and one column per digit.

    # Pick the digit with the highest score in each row -
    # The highest score in column d + 1 means digit d. eachrow(S) goes through the
    # rows of a matrix S, and argmax(v) gives the position of the largest entry of
    # a vector v (the first one, on a tie).

    error("Complete predict_digits in src/Compute.jl");
end


# ----------------------------------------------------------------------------
# Part 2b: Count the mix-ups
# ----------------------------------------------------------------------------

"""
    confusion_table(actual::AbstractVector{<:Integer},
        predicted::AbstractVector{<:Integer}) -> Matrix{Int}

Count how often each true digit was read as each digit.

# Arguments

- `actual`: the true digit of each image, from 0 to 9.
- `predicted`: the digit the model read for each image, in the same order.

# Returns

A new 10 × 10 `Matrix{Int}`, called a confusion table. Rows are the true digits and
columns are the digits the model read. Entry `[a + 1, p + 1]` counts the images
whose true digit is `a` and that the model read as `p`. The diagonal counts the
correct readings, and each entry off the diagonal counts one kind of mix-up.
Empty inputs give a table of zeros. Leave `actual` and `predicted` unchanged.

# Errors

Throw `DimensionMismatch` if `actual` and `predicted` have different lengths, and
`ArgumentError` if any value in either one is outside 0 to 9.
"""
function confusion_table(actual::AbstractVector{<:Integer}, predicted::AbstractVector{<:Integer})

    # TODO 4: Build and return the 10 × 10 confusion table.

    # Check the inputs -
    # Check the lengths, then check that no value is outside 0 to 9.

    # Count each pair (true digit, digit read) -
    # Start from a 10 × 10 table of zeros. For each image, add one to row a + 1
    # and column p + 1, where a is its true digit and p is the digit read.

    # Return the table -

    error("Complete confusion_table in src/Compute.jl");
end
