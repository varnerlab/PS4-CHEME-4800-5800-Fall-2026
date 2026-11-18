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

toy = toy_digits(); # 22 tiny "images" with three pixels each


# ----------------------------------------------------------------------------
# Run the 24 scored checks | One labeled case, one @test each
# ----------------------------------------------------------------------------
# A case whose code raises an error counts as one failed test.

@testset "PS4 Part 1 (24 tests)" begin

    # Yes/no labels for one digit -
    @testset "labels type and length" begin
        y = digit_labels([3, 1, 3, 0, 7], 3);
        @test y isa Array{Float64,1} && length(y) == 5;
    end

    @testset "plus one exactly where the label matches" begin
        @test digit_labels([3, 1, 3, 0], 3) == [1.0, -1.0, 1.0, -1.0];
    end

    @testset "digit zero" begin
        @test digit_labels([0, 5, 0], 0) == [1.0, -1.0, 1.0];
    end

    @testset "digit nine" begin
        @test digit_labels([9, 9, 1], 9) == [1.0, 1.0, -1.0];
    end

    @testset "no matching label gives all minus one" begin
        @test digit_labels([1, 2, 3], 7) == fill(-1.0, 3);
    end

    @testset "real labels: counts match" begin
        labels = small_training(1000).labels;
        matches = true;
        for d in 0:9
            y = digit_labels(labels, d);
            if count(y .== 1.0) != count(labels .== d) # one 1.0 per image of digit d
                matches = false;
            end
        end
        @test matches;
    end

    @testset "empty labels give an empty vector" begin
        y = digit_labels(Int[], 4);
        @test y isa Array{Float64,1} && isempty(y);
    end

    @testset "labels unchanged" begin
        labels = [4, 2, 4, 9];
        digit_labels(labels, 4);
        @test labels == [4, 2, 4, 9];
    end

    @testset "digit outside 0 to 9 rejected" begin

        # Both calls must throw an ArgumentError -
        rejects_10 = false;
        try
            digit_labels([1, 2], 10);
        catch caught
            rejects_10 = caught isa ArgumentError;
        end
        rejects_minus_1 = false;
        try
            digit_labels([1, 2], -1);
        catch caught
            rejects_minus_1 = caught isa ArgumentError;
        end
        @test rejects_10 && rejects_minus_1;
    end

    @testset "label outside 0 to 9 rejected" begin
        @test_throws ArgumentError digit_labels([1, 12], 1);
    end

    # One model per digit -
    @testset "parameter matrix size" begin
        @test size(train_digit_models(toy.X, toy.labels; maxiter=5)) == (4, 10);
    end

    @testset "parameter matrix type" begin
        @test train_digit_models(toy.X, toy.labels; maxiter=5) isa Array{Float64,2};
    end

    @testset "logistic column d + 1 is the model for digit d" begin
        W = train_digit_models(toy.X, toy.labels; method=:logistic, maxiter=50);
        expected = reference_models(toy.X, toy.labels; method=:logistic, maxiter=50);
        @test isapprox(W, expected; rtol=1e-8);
    end

    @testset "Perceptron column d + 1 is the model for digit d" begin
        W = train_digit_models(toy.X, toy.labels; method=:perceptron, maxiter=10);
        expected = reference_models(toy.X, toy.labels; method=:perceptron, maxiter=10);
        @test isapprox(W, expected; rtol=1e-8);
    end

    @testset "first and last columns are digits 0 and 9" begin
        W = train_digit_models(toy.X, toy.labels; maxiter=50);
        α = 1 / opnorm(toy.X)^2;
        θ0 = my_logistic_regression(toy.X, reference_labels(toy.labels, 0); α=α, maxiter=50).θ;
        θ9 = my_logistic_regression(toy.X, reference_labels(toy.labels, 9); α=α, maxiter=50).θ;
        @test isapprox(W[:, 1], θ0; rtol=1e-8) && isapprox(W[:, 10], θ9; rtol=1e-8) && W[:, 1] != W[:, 10];
    end

    @testset "logistic is the default method" begin
        W_default = train_digit_models(toy.X, toy.labels; maxiter=20);
        W_logistic = train_digit_models(toy.X, toy.labels; method=:logistic, maxiter=20);
        @test W_default == W_logistic;
    end

    @testset "maxiter reaches logistic regression" begin

        # One step must match one reference step, and differ from 50 reference steps -
        W1 = train_digit_models(toy.X, toy.labels; method=:logistic, maxiter=1);
        one_step = reference_models(toy.X, toy.labels; method=:logistic, maxiter=1);
        fifty_steps = reference_models(toy.X, toy.labels; method=:logistic, maxiter=50);
        @test isapprox(W1, one_step; rtol=1e-8) && !isapprox(W1, fifty_steps; rtol=1e-3);
    end

    @testset "maxiter reaches the Perceptron" begin

        # One pass must match one reference pass, and differ from 10 reference passes -
        W1 = train_digit_models(toy.X, toy.labels; method=:perceptron, maxiter=1);
        one_pass = reference_models(toy.X, toy.labels; method=:perceptron, maxiter=1);
        ten_passes = reference_models(toy.X, toy.labels; method=:perceptron, maxiter=10);
        @test isapprox(W1, one_pass; rtol=1e-8) && W1 != ten_passes;
    end

    @testset "real images, logistic" begin
        data = small_training(500);
        W = train_digit_models(data.X, data.labels; method=:logistic, maxiter=20);
        expected = reference_models(data.X, data.labels; method=:logistic, maxiter=20);
        @test isapprox(W, expected; rtol=1e-8);
    end

    @testset "real images, Perceptron" begin
        data = small_training(500);
        W = train_digit_models(data.X, data.labels; method=:perceptron, maxiter=2);
        expected = reference_models(data.X, data.labels; method=:perceptron, maxiter=2);
        @test isapprox(W, expected; rtol=1e-8);
    end

    @testset "inputs unchanged" begin
        X = copy(toy.X);
        labels = copy(toy.labels);
        train_digit_models(X, labels; method=:logistic, maxiter=5);
        train_digit_models(X, labels; method=:perceptron, maxiter=5);
        @test X == toy.X && labels == toy.labels;
    end

    @testset "a digit with no images still gets a column" begin
        labels = replace(toy.labels, 5 => 4); # every 5 becomes a 4, so no image is a 5
        W = train_digit_models(toy.X, labels; maxiter=30);
        all_minus_one = fill(-1.0, length(labels)); # the labels for digit 5: no image is a 5
        θ5 = my_logistic_regression(toy.X, all_minus_one; α=1/opnorm(toy.X)^2, maxiter=30).θ;
        @test size(W) == (4, 10) && isapprox(W[:, 6], θ5; rtol=1e-8);
    end

    @testset "unknown method rejected" begin
        @test_throws ArgumentError train_digit_models(toy.X, toy.labels; method=:other, maxiter=5);
    end

    @testset "labels length must match X" begin
        @test_throws DimensionMismatch train_digit_models(toy.X, toy.labels[1:end-1]; maxiter=5);
    end
end;
