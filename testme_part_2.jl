# Run: julia --startup-file=no --project=. testme_part_2.jl
# Add --solution to test solution/src/Compute.jl without replacing student code.
# Each labeled case checks one requirement of predict_digits or confusion_matrix.
# These tests do not call the Part 1 functions.


# ----------------------------------------------------------------------------
# Setup | Load the selected implementation and test helpers
# ----------------------------------------------------------------------------

using Test; # record one scored assertion per labeled case

Base.include(@__MODULE__, joinpath(@__DIR__, "Include.jl"));
Base.include(@__MODULE__, joinpath(@__DIR__, "test_support.jl"));

using .PS4Checks; # small inputs and independent reference results

"""Return a 3 × 10 parameter matrix whose only nonzero column is digit `d`."""
function only_digit(d)

    W = zeros(3, 10);
    W[:, d + 1] .= 1.0; # column d + 1 holds digit d

    return W;
end

# Each case contributes one scored test, including cases whose implementation raises an error.
X3 = [1.0 2.0 1.0; 0.5 0.1 1.0]; # two "images": two pixels and the bias

# ----------------------------------------------------------------------------
# Test cases | One scored assertion per labeled case
# ----------------------------------------------------------------------------

checks = [
    # Read each image as the digit with the highest score -
    "predictions type and length" => () -> begin
        p = predict_digits(only_digit(2), X3);

        p isa Array{Int64,1} && length(p) == 2
    end,

    "highest score wins" => () -> begin
        W = zeros(3, 10);
        W[:, 5] = [1.0, 0.0, 0.0]; # digit 4 scores 1.0 and 0.5
        W[:, 8] = [0.0, 0.0, 0.8]; # digit 7 scores 0.8 and 0.8

        predict_digits(W, X3) == [4, 7]
    end,

    "column 1 is digit 0 and column 10 is digit 9" => () ->
        predict_digits(only_digit(0), X3) == [0, 0] &&
        predict_digits(only_digit(9), X3) == [9, 9],

    "every digit can be predicted" => () -> begin
        X = zeros(10, 11); # ten images with ten pixels each, then the bias
        W = zeros(11, 10);
        for k in 1:10
            X[k, k] = 1.0;  # image k has ink only in pixel k
            X[k, 11] = 1.0; # the 1 for the bias
            W[k, k] = 1.0;  # the model in column k scores only pixel k
        end

        predict_digits(W, X) == collect(0:9)
    end,

    "ties go to the smallest digit" => () -> begin
        W = zeros(3, 10);
        W[:, 4] .= 2.0; # digits 3 and 7 tie for the highest score
        W[:, 8] .= 2.0;

        predict_digits(W, X3) == [3, 3] && predict_digits(zeros(3, 10), X3) == [0, 0]
    end,

    "all scores negative" => () -> begin
        W = fill(-5.0, 3, 10);
        W[:, 7] .= -1.0; # digit 6 is the least negative

        predict_digits(W, X3) == [6, 6]
    end,

    "one image" => () ->
        predict_digits(only_digit(5), X3[1:1, :]) == [5],

    "zero images" => () -> begin
        p = predict_digits(only_digit(5), zeros(0, 3));

        p isa Array{Int64,1} && isempty(p)
    end,

    "real images with reference parameters" => () -> begin
        data = small_training(300);
        W = reference_models(data.X, data.labels; method=:logistic, maxiter=20);
        X = testing_images(200).X;
        scores = X * W;
        expected = zeros(Int, 200);
        for i in 1:200
            expected[i] = argmax(scores[i, :]) - 1; # column d + 1 holds digit d
        end

        predict_digits(W, X) == expected
    end,

    "inputs unchanged" => () -> begin
        W = only_digit(3);
        X = copy(X3);
        predict_digits(W, X);

        W == only_digit(3) && X == X3
    end,

    "W with the wrong number of rows rejected" => () ->
        throws_type(() -> predict_digits(zeros(4, 10), X3), DimensionMismatch),

    "W with the wrong number of columns rejected" => () ->
        throws_type(() -> predict_digits(zeros(3, 9), X3), DimensionMismatch),

    # Count which digit each true digit was read as -
    "matrix type and size" => () -> begin
        CM = confusion_matrix([1, 2], [1, 3]);

        CM isa Array{Int64,2} && size(CM) == (10, 10)
    end,

    "hand example" => () -> begin
        expected = zeros(Int, 10, 10);
        expected[1, 1] = 1;   # a 0 read as 0
        expected[2, 8] = 1;   # a 1 read as 7
        expected[2, 2] = 1;   # a 1 read as 1
        expected[10, 10] = 1; # a 9 read as 9

        confusion_matrix([0, 1, 1, 9], [0, 7, 1, 9]) == expected
    end,

    "rows are true digits, columns are readings" => () -> begin
        CM = confusion_matrix([4], [9]);

        CM[5, 10] == 1 && CM[10, 5] == 0 && sum(CM) == 1
    end,

    "perfect readings lie on the diagonal" => () -> begin
        labels = testing_images(500).labels;
        CM = confusion_matrix(labels, labels);
        diagonal = 0;
        for k in 1:10
            diagonal += CM[k, k];
        end

        sum(CM) == diagonal
    end,

    "total equals the number of images" => () ->
        sum(confusion_matrix([0, 3, 3, 8, 8, 8], [0, 3, 5, 8, 3, 1])) == 6,

    "row sums count each true digit" => () -> begin
        actual = [0, 3, 3, 8, 8, 8];
        predicted = [0, 3, 5, 8, 3, 1];
        rows = vec(sum(confusion_matrix(actual, predicted); dims=2)); # the sum of each row
        expected = zeros(Int, 10);
        for d in 0:9
            expected[d + 1] = count(actual .== d); # images whose true digit is d
        end

        rows == expected
    end,

    "column sums count each reading" => () -> begin
        actual = [0, 3, 3, 8, 8, 8];
        predicted = [0, 3, 5, 8, 3, 1];
        columns = vec(sum(confusion_matrix(actual, predicted); dims=1)); # the sum of each column
        expected = zeros(Int, 10);
        for d in 0:9
            expected[d + 1] = count(predicted .== d); # images read as d
        end

        columns == expected
    end,

    "empty input gives all zeros" => () ->
        confusion_matrix(Int[], Int[]) == zeros(Int, 10, 10),

    "real labels against themselves" => () -> begin
        labels = testing_images(10_000).labels;
        CM = confusion_matrix(labels, labels);
        matches = true;
        for d in 0:9
            if CM[d + 1, d + 1] != count(labels .== d) # every image of d read as d
                matches = false;
            end
        end

        matches
    end,

    "inputs unchanged" => () -> begin
        actual = [2, 2, 7];
        predicted = [2, 1, 7];
        confusion_matrix(actual, predicted);

        actual == [2, 2, 7] && predicted == [2, 1, 7]
    end,

    "different lengths rejected" => () ->
        throws_type(() -> confusion_matrix([1, 2, 3], [1, 2]), DimensionMismatch),

    "values outside 0 to 9 rejected" => () ->
        throws_type(() -> confusion_matrix([10, 2], [1, 2]), ArgumentError) &&
        throws_type(() -> confusion_matrix([1, 2], [1, -1]), ArgumentError),
];


# ----------------------------------------------------------------------------
# Run the 24 scored checks
# ----------------------------------------------------------------------------

@testset "PS4 Part 2 (24 tests)" begin
    for (label, check) in checks
        @testset "$label" begin
            @test check();
        end
    end
end;
