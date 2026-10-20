# Supplied loader and trainer checks, separate from the 48 scored student tests.
# Run: julia --startup-file=no --project=. instructor/test_trainers.jl
using Test;          # loader and trainer assertions
using LinearAlgebra; # opnorm for the L9d step size
include(joinpath(@__DIR__, "..", "Include.jl"));

# The L9b and L9d reference solutions, each loaded into its own module -
const LABS = joinpath(@__DIR__, "..", "..", "CHEME-5800-CourseRepository-Fall-2026", "weeks", "week-09");
const LAB_PERCEPTRON = joinpath(LABS, "L9b", "src", "Compute-solution.jl");
const LAB_LOGISTIC = joinpath(LABS, "L9d", "src", "Compute-solution.jl");
has_labs = isfile(LAB_PERCEPTRON) && isfile(LAB_LOGISTIC);
has_labs || @warn "Course repository not found next to PS4; skipping the comparisons with the L9b and L9d labs.";
module LabB end
module LabD end
if has_labs
    Base.include(LabB, LAB_PERCEPTRON);
    Base.include(LabD, LAB_LOGISTIC);
end

@testset "Supplied loader and trainers" begin
    d = load_digits();

    # Loader: sizes, bias column, pixel range, and labels -
    for set in (d.training, d.test)
        @test size(set.X) == (10_000, 785);
        @test set.X isa Matrix{Float64};
        @test all(set.X[:, end] .== 1.0);
        @test all(0.0 .<= set.X[:, 1:784] .<= 1.0);
        @test eltype(set.labels) == Int && length(set.labels) == 10_000;
        @test all(0 .<= set.labels .<= 9);
        @test all(count(==(digit), set.labels) >= 850 for digit in 0:9);
    end

    # Loader: the first training image is the official 5, upright -
    @test d.training.labels[1] == 5;
    picture = permutedims(reshape(d.training.X[1, 1:784], 28, 28));
    ink_rows = findall(r -> any(picture[r, :] .> 0.5), 1:28);
    @test first(ink_rows) == 6 && last(ink_rows) == 25;
    # The top bar of this 5 runs to the right, its bottom curve ends at the left -
    @test picture[7, 20] > 0.5 && picture[25, 6] > 0.5 && picture[25, 20] < 0.5;

    # Loader: a cut-short file gives a clear error naming the file and its size -
    folder = mktempdir();
    for name in readdir(joinpath(@__DIR__, "..", "data"))
        cp(joinpath(@__DIR__, "..", "data", name), joinpath(folder, name));
    end
    write(joinpath(folder, "train-images.bin"), read(joinpath(folder, "train-images.bin"))[1:1000]);
    message = try
        load_digits(folder);
        "";
    catch caught
        sprint(showerror, caught);
    end
    @test occursin("train-images.bin", message) && occursin("7840000", message);

    # Perceptron: same results as the L9b lab on a small separable set -
    X = [1.0 1.0 1.0; 1.0 -1.0 1.0; -1.0 1.0 1.0; -1.0 -1.0 1.0; 2.0 0.5 1.0];
    y = [1.0, 1.0, -1.0, -1.0, 1.0];
    ours = my_perceptron(X, y; maxiter=100);
    @test ours.mistakes == 0;
    if has_labs
        lab = LabB.L9bXOR.my_perceptron(X, y; maxiter=100);
        @test ours.θ == lab.θ && ours.passes == lab.passes && ours.mistakes == lab.mistakes;
        lab3 = LabB.L9bXOR.my_perceptron(d.training.X[1:200, :], [l == 3 ? 1.0 : -1.0 for l in d.training.labels[1:200]]; maxiter=3);
        ours3 = my_perceptron(d.training.X[1:200, :], [l == 3 ? 1.0 : -1.0 for l in d.training.labels[1:200]]; maxiter=3);
        @test ours3.θ == lab3.θ && ours3.passes == lab3.passes && ours3.mistakes == lab3.mistakes;
    end

    # Logistic regression: same results as the L9d lab on 50 real images -
    X50 = d.training.X[1:50, :];
    y50 = [l == 3 ? 1.0 : -1.0 for l in d.training.labels[1:50]];
    α = 1 / opnorm(X50)^2;
    ours = my_logistic_regression(X50, y50; α=α, maxiter=25);
    @test length(ours.θ) == 785 && ours.iterations == 25 && !ours.converged;
    @test length(ours.losses) == 26 && all(diff(ours.losses) .<= 1e-12);
    if has_labs
        lab = LabD.L9dLogistic.my_logistic_regression(X50, y50; α=α, maxiter=25);
        @test isapprox(ours.θ, lab.θ; rtol=1e-10);
        @test ours.iterations == lab.iterations && ours.converged == lab.converged;
        @test isapprox(ours.losses, lab.losses; rtol=1e-10);
        tiny = [1.0 1.0; -1.0 1.0];
        @test my_logistic_regression(tiny, [1.0, -1.0]; α=0.1, ϵ=1e-3).iterations ==
              LabD.L9dLogistic.my_logistic_regression(tiny, [1.0, -1.0]; α=0.1, ϵ=1e-3).iterations;
    end

    # Both trainers reject bad labels and mismatched lengths -
    @test_throws ArgumentError my_perceptron(X, [1.0, 0.0, 1.0, -1.0, 1.0]);
    @test_throws DimensionMismatch my_perceptron(X, [1.0, -1.0]);
    @test_throws ArgumentError my_logistic_regression(X, [1.0, 0.0, 1.0, -1.0, 1.0]; α=0.1);
    @test_throws DimensionMismatch my_logistic_regression(X, [1.0, -1.0]; α=0.1);
end
