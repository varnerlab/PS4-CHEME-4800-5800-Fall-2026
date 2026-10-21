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

    # Every digit twice, then one more 5 and one more 3 -
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 5, 3];
    n = length(labels);

    # Fill three pixels per image from a fixed formula; the last column stays 1 for the bias -
    X = ones(n, 4);
    for i in 1:n
        for j in 1:3
            X[i, j] = mod(0.37 * i + 0.61 * j * i + 0.13 * labels[i] * j, 1.0);
        end
    end

    return (X = X, labels = labels);
end


"""
    small_training(n) -> NamedTuple

Return `(X, labels)` with new copies of the first `n` training images and labels.
"""
function small_training(n)

    data = load_digits();
    X = data.training.X[1:n, :];
    labels = data.training.labels[1:n];

    return (X = X, labels = labels);
end


"""
    testing_images(n) -> NamedTuple

Return `(X, labels)` with new copies of the first `n` testing images and labels.
"""
function testing_images(n)

    data = load_digits();
    X = data.testing.X[1:n, :];
    labels = data.testing.labels[1:n];

    return (X = X, labels = labels);
end


# ----------------------------------------------------------------------------
# Reference results
# ----------------------------------------------------------------------------

"""
    reference_labels(labels, d) -> Array{Float64,1}

Return 1.0 where `labels` equals `d` and -1.0 elsewhere, without checking inputs.
"""
function reference_labels(labels, d)

    y = zeros(length(labels));
    for i in 1:length(labels)
        if labels[i] == d
            y[i] = 1.0;
        else
            y[i] = -1.0;
        end
    end

    return y;
end


"""
    reference_models(X, labels; method=:logistic, maxiter) -> Array{Float64,2}

Train one model per digit with direct calls to the training functions and return
the `size(X, 2)` × 10 matrix with the model for digit `d` in column `d + 1`.
Logistic regression uses `α = 1/opnorm(X)^2`, as the README specifies.
"""
function reference_models(X, labels; method=:logistic, maxiter)

    α = 1 / opnorm(X)^2; # the L9d learning rate
    W = zeros(size(X, 2), 10);
    for d in 0:9
        y = reference_labels(labels, d);
        if method == :logistic
            W[:, d + 1] = my_logistic_regression(X, y; α = α, maxiter = maxiter).θ;
        else
            W[:, d + 1] = my_perceptron(X, y; maxiter = maxiter).θ;
        end
    end

    return W;
end


"""
    throws_type(f, T) -> Bool

Call the zero-argument function `f`. Return true only if it throws an exception
of type `T`. Return false if no exception is thrown or the type differs.
"""
function throws_type(f, T)

    # Run f; if it throws, report whether the error has the expected type -
    try
        f();
    catch e
        return e isa T;
    end

    # f finished without an error -
    return false;
end

end
