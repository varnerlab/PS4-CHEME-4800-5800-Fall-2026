# Load the assignment from its own directory, independent of the working directory.
"""
    PlantLogbook

Load the data loader, the two training functions, and the four student functions.
`load_digits()` returns the training and testing images. `my_perceptron` and
`my_logistic_regression` are the yes/no training functions from the L9b and L9d
labs. The student functions and the reference solution use the same loader and
training functions.
Pass `--solution` after a script's filename to load `solution/src/Compute.jl`
instead of the student file. The local solution must exist; no files are copied
or replaced. Without that flag, always load `src/Compute.jl`.
Include this file once per Julia session to make the exported names available.
Restart Julia after editing source files, or run the scripts in a new process.
"""
module PlantLogbook

using LinearAlgebra # dot, norm, and opnorm

# Load the data loader and training functions before the student functions that use them -
include(joinpath(@__DIR__, "src", "Data.jl"));
include(joinpath(@__DIR__, "src", "Trainers.jl"));
# If you add helper files, include them here in dependency order before Compute.jl.
# For example, after creating src/MyHelpers.jl:
# include(joinpath(@__DIR__, "src", "MyHelpers.jl"));
# Select exactly one implementation; never fall back when the solution is absent -
const COMPUTE_PATH = "--solution" in ARGS ?
    joinpath(@__DIR__, "solution", "src", "Compute.jl") :
    joinpath(@__DIR__, "src", "Compute.jl");
isfile(COMPUTE_PATH) || error("Implementation file is missing: $COMPUTE_PATH. --solution requires the local instructor solution.");
include(COMPUTE_PATH);

export load_digits, my_perceptron, my_logistic_regression;
export digit_labels, train_digit_models, predict_digits, confusion_matrix;

end

using .PlantLogbook; # make the assignment functions available to the calling script
