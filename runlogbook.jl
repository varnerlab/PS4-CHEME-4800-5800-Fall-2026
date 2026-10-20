# ----------------------------------------------------------------------------
# Train both digit readers and write the reports
# ----------------------------------------------------------------------------
# Run from the assignment root:
#   julia --startup-file=no --project=. runlogbook.jl
#   julia --startup-file=no --project=. runlogbook.jl --solution
#
# The --solution flag loads the reference code and writes solution/outputs/.
#
# After Parts 1 and 2 are complete, train logistic regression and the Perceptron
# on the training images, read the testing images, and write the plots and tables
# to outputs/. Existing files with the same names are overwritten.


# 1. Load the implementation and reporting utilities -
ENV["GKSwstype"] = "100"; # render plots to files without opening a GUI window

include(joinpath(@__DIR__, "Include.jl"));
include(joinpath(@__DIR__, "src", "Reporting.jl"));


# 2. Select the output directory -
output_directory = "--solution" in ARGS ?
    joinpath(@__DIR__, "solution", "outputs") : joinpath(@__DIR__, "outputs");

"--solution" in ARGS && println("Running the instructor reference solution.");


# 3. Train, read the testing images, and write the reports -
try
    LogbookReports.write_outputs(PlantLogbook, output_directory);
catch caught
    # An unfinished function gets a short message instead of a long error listing -
    if caught isa ErrorException && startswith(caught.msg, "Complete ")
        println(stderr, "Cannot make the reports yet: ", caught.msg, ".");
        println(stderr, "Finish Parts 1 and 2 and pass testme_part_1.jl and testme_part_2.jl first.");
        isinteractive() || exit(1); # keep an open REPL open
    else
        rethrow();
    end
end
