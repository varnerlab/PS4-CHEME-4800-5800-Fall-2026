# Run: julia --startup-file=no --project=. testme_part_1.jl
# Add --solution to test solution/src/Compute.jl without replacing student code.
# Each labeled case checks one requirement of digit_labels or train_digit_models.


# ----------------------------------------------------------------------------
# Setup | Load the selected implementation and test helpers
# ----------------------------------------------------------------------------

using Test;          # record one scored assertion per labeled case
using LinearAlgebra; # opnorm for the logistic-regression learning rate

Base.include(@__MODULE__, joinpath(@__DIR__, "Include.jl"));
Base.include(@__MODULE__, joinpath(@__DIR__, "test_support.jl"));

using .PS4Checks; # small inputs and independent reference results

# Each case contributes one scored test, including cases whose implementation raises an error.
toy = toy_digits();

# ----------------------------------------------------------------------------
# Test cases | One scored assertion per labeled case
# ----------------------------------------------------------------------------

checks = [
    # Yes/no labels for one digit -
    "labels type and length" => () -> begin
        y = digit_labels([3, 1, 3, 0, 7], 3);

        y isa Array{Float64,1} && length(y) == 5
    end,

    "plus one exactly where the label matches" => () ->
        digit_labels([3, 1, 3, 0], 3) == [1.0, -1.0, 1.0, -1.0],

    "digit zero" => () ->
        digit_labels([0, 5, 0], 0) == [1.0, -1.0, 1.0],

    "digit nine" => () ->
        digit_labels([9, 9, 1], 9) == [1.0, 1.0, -1.0],

    "no matching label gives all minus one" => () ->
        digit_labels([1, 2, 3], 7) == fill(-1.0, 3),

    "real labels: counts match" => () -> begin
        labels = small_training(1000).labels;
        matches = true;
        for d in 0:9
            y = digit_labels(labels, d);
            if count(y .== 1.0) != count(labels .== d) # one 1.0 per image of digit d
                matches = false;
            end
        end

        matches
    end,

    "empty labels give an empty vector" => () -> begin
        y = digit_labels(Int[], 4);

        y isa Array{Float64,1} && isempty(y)
    end,

    "labels unchanged" => () -> begin
        labels = [4, 2, 4, 9];
        digit_labels(labels, 4);

        labels == [4, 2, 4, 9]
    end,

    "digit outside 0 to 9 rejected" => () ->
        throws_type(() -> digit_labels([1, 2], 10), ArgumentError) &&
        throws_type(() -> digit_labels([1, 2], -1), ArgumentError),

    "label outside 0 to 9 rejected" => () ->
        throws_type(() -> digit_labels([1, 12], 1), ArgumentError),

    # One model per digit -
    "parameter matrix size" => () ->
        size(train_digit_models(toy.X, toy.labels; maxiter=5)) == (4, 10),

    "parameter matrix type" => () ->
        train_digit_models(toy.X, toy.labels; maxiter=5) isa Array{Float64,2},

    "logistic column d + 1 is the model for digit d" => () -> isapprox(
        train_digit_models(toy.X, toy.labels; method=:logistic, maxiter=50),
        reference_models(toy.X, toy.labels; method=:logistic, maxiter=50); rtol=1e-8),

    "Perceptron column d + 1 is the model for digit d" => () -> isapprox(
        train_digit_models(toy.X, toy.labels; method=:perceptron, maxiter=10),
        reference_models(toy.X, toy.labels; method=:perceptron, maxiter=10); rtol=1e-8),

    "first and last columns are digits 0 and 9" => () -> begin
        W = train_digit_models(toy.X, toy.labels; maxiter=50);
        α = 1 / opnorm(toy.X)^2;
        θ0 = my_logistic_regression(toy.X, reference_labels(toy.labels, 0); α=α, maxiter=50).θ;
        θ9 = my_logistic_regression(toy.X, reference_labels(toy.labels, 9); α=α, maxiter=50).θ;

        isapprox(W[:, 1], θ0; rtol=1e-8) && isapprox(W[:, 10], θ9; rtol=1e-8) && W[:, 1] != W[:, 10]
    end,

    "logistic is the default method" => () ->
        train_digit_models(toy.X, toy.labels; maxiter=20) ==
        train_digit_models(toy.X, toy.labels; method=:logistic, maxiter=20),

    "maxiter reaches logistic regression" => () -> begin
        W1 = train_digit_models(toy.X, toy.labels; method=:logistic, maxiter=1);

        isapprox(W1, reference_models(toy.X, toy.labels; method=:logistic, maxiter=1); rtol=1e-8) &&
            !isapprox(W1, reference_models(toy.X, toy.labels; method=:logistic, maxiter=50); rtol=1e-3)
    end,

    "maxiter reaches the Perceptron" => () -> begin
        W1 = train_digit_models(toy.X, toy.labels; method=:perceptron, maxiter=1);

        isapprox(W1, reference_models(toy.X, toy.labels; method=:perceptron, maxiter=1); rtol=1e-8) &&
            W1 != reference_models(toy.X, toy.labels; method=:perceptron, maxiter=10)
    end,

    "real images, logistic" => () -> begin
        data = small_training(500);

        isapprox(train_digit_models(data.X, data.labels; method=:logistic, maxiter=20),
            reference_models(data.X, data.labels; method=:logistic, maxiter=20); rtol=1e-8)
    end,

    "real images, Perceptron" => () -> begin
        data = small_training(500);

        isapprox(train_digit_models(data.X, data.labels; method=:perceptron, maxiter=2),
            reference_models(data.X, data.labels; method=:perceptron, maxiter=2); rtol=1e-8)
    end,

    "inputs unchanged" => () -> begin
        X = copy(toy.X);
        labels = copy(toy.labels);
        train_digit_models(X, labels; method=:logistic, maxiter=5);
        train_digit_models(X, labels; method=:perceptron, maxiter=5);

        X == toy.X && labels == toy.labels
    end,

    "a digit with no images still gets a column" => () -> begin
        labels = replace(toy.labels, 5 => 4); # every 5 becomes a 4, so no image is a 5
        W = train_digit_models(toy.X, labels; maxiter=30);
        θ5 = my_logistic_regression(toy.X, fill(-1.0, length(labels)); α=1/opnorm(toy.X)^2, maxiter=30).θ;

        size(W) == (4, 10) && isapprox(W[:, 6], θ5; rtol=1e-8)
    end,

    "unknown method rejected" => () ->
        throws_type(() -> train_digit_models(toy.X, toy.labels; method=:other, maxiter=5), ArgumentError),

    "labels length must match X" => () ->
        throws_type(() -> train_digit_models(toy.X, toy.labels[1:end-1]; maxiter=5), DimensionMismatch),
];


# ----------------------------------------------------------------------------
# Run the 24 scored checks
# ----------------------------------------------------------------------------

@testset "PS4 Part 1 (24 tests)" begin
    for (label, check) in checks
        @testset "$label" begin
            @test check();
        end
    end
end;
