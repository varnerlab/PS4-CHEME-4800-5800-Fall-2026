# ----------------------------------------------------------------------------
# Supplied test support | Small inputs and independent reference results
# ----------------------------------------------------------------------------

"""Supplied test inputs and reference calculations for the PS4 student functions."""
module PS4Checks

using LinearAlgebra; # opnorm for the logistic-regression learning rate

using ..PlantLogbook; # implementation loaded by the current test suite

export throws_type, reference_labels, toy_digits, small_training, testing_images, reference_models;

# ----------------------------------------------------------------------------
# Test inputs
# ----------------------------------------------------------------------------

"""
    toy_digits() -> NamedTuple

Build a tiny set of 22 "images" with three pixels each, plus the bias column.
Return `(X, labels)`: `X` is a new 22 × 4 `Array{Float64,2}` and `labels` a new
`Array{Int64,1}` that holds every digit from 0 to 9 at least twice. The pixel values
follow a fixed formula, so the ten trained models differ from each other.
"""
function toy_digits()

    labels = [collect(0:9); collect(0:9); 5; 3];
    n = length(labels);
    pixels = [mod(0.37 * i + 0.61 * j * i + 0.13 * labels[i] * j, 1.0) for i in 1:n, j in 1:3];

    return (X = [pixels ones(n)], labels = labels);
end


const DIGITS = Ref{Any}(nothing); # load_digits() result, read once per test suite

"""
    small_training(n) -> NamedTuple

Return `(X, labels)` with new copies of the first `n` training images and labels.
"""
function small_training(n)

    isnothing(DIGITS[]) && (DIGITS[] = load_digits());
    data = DIGITS[].training;

    return (X = data.X[1:n, :], labels = data.labels[1:n]);
end


"""
    testing_images(n) -> NamedTuple

Return `(X, labels)` with new copies of the first `n` testing images and labels.
"""
function testing_images(n)

    isnothing(DIGITS[]) && (DIGITS[] = load_digits());
    data = DIGITS[].testing;

    return (X = data.X[1:n, :], labels = data.labels[1:n]);
end


# ----------------------------------------------------------------------------
# Reference results
# ----------------------------------------------------------------------------

"""
    reference_labels(labels, d) -> Array{Float64,1}

Return 1.0 where `labels` equals `d` and -1.0 elsewhere, without checking inputs.
"""
reference_labels(labels, d) = [label == d ? 1.0 : -1.0 for label in labels];


"""
    reference_models(X, labels; method=:logistic, maxiter) -> Array{Float64,2}

Train one model per digit with direct calls to the training functions and return
the `size(X, 2)` × 10 matrix with the model for digit `d` in column `d + 1`.
Logistic regression uses `α = 1/opnorm(X)^2`, as the README specifies.
"""
function reference_models(X, labels; method=:logistic, maxiter)

    α = 1 / opnorm(X)^2;
    W = zeros(size(X, 2), 10);
    for d in 0:9
        y = reference_labels(labels, d);
        W[:, d + 1] = method == :logistic ?
            my_logistic_regression(X, y; α = α, maxiter = maxiter).θ :
            my_perceptron(X, y; maxiter = maxiter).θ;
    end

    return W;
end


"""
    throws_type(f, T) -> Bool

Call the zero-argument function `f`. Return true only if it throws an exception
of type `T`. Return false if no exception is thrown or the type differs.
"""
function throws_type(f, T)

    try
        f();
    catch e
        return e isa T;
    end

    return false;
end

end
