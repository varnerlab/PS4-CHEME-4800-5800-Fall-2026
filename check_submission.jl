# Check required work, generate outputs, and write a manifest.
# Usage: julia --startup-file=no --project=. check_submission.jl
# Add --solution to check solution/src/Compute.jl and solution/responses.md; the
# outputs and manifest then go under solution/. Your own files stay unchanged.
# Exit with code 0 when the tests, output generation, and response checks pass;
# otherwise exit with code 1. A score still requires instructor review.
using Test;  # distinguish failed assertions from source-loading errors
using SHA;   # fingerprint submitted source files and responses
using Dates; # timestamp the submission manifest

# Choose the files being checked before loading any suite or writing outputs -
const SOLUTION_MODE = "--solution" in ARGS;
if SOLUTION_MODE
    const CHECK_ROOT = joinpath(@__DIR__, "solution");
else
    const CHECK_ROOT = @__DIR__;
end
const RESPONSE_PATH = joinpath(CHECK_ROOT, "responses.md");
const COMPUTE_PATH = joinpath(CHECK_ROOT, "src", "Compute.jl");
if SOLUTION_MODE
    for path in (COMPUTE_PATH, RESPONSE_PATH)
        if !isfile(path)
            error("--solution requires the local instructor file: $path");
        end
    end
    println("Checking the instructor reference solution; outputs and manifest go under solution/.");
end

"""
    rootcause(caught)

Inspect a caught exception and return the first exception that is not a
`LoadError`, unwrapping nested file-include errors. Return `caught` itself if
it has no wrapper. Leave the exception unchanged. The checker uses this result
to distinguish failed test assertions from errors that prevented a suite running.
"""
function rootcause(caught)

    # A failed include wraps the real error in a LoadError; unwrap each layer -
    while caught isa LoadError
        caught = caught.error;
    end

    return caught;
end

"""
    discussion_issues(path; questions=1:3) -> Vector{String}

Read the response file at `path` and return messages for missing or duplicate
questions, empty answers, and remaining TODO/TBD placeholders for `questions`.
Expect numbered, bold question prompts followed by answer paragraphs in responses.md.
Ignore HTML comments when checking whether an answer is present.

Return an empty vector if no issues are detected; this does not grade correctness.
A missing or unreadable file returns a warning message. The file is unchanged.
"""
function discussion_issues(path; questions=1:3)

    # Read the response file -
    if !isfile(path)
        return ["responses.md is missing"];
    end
    content = "";
    try
        content = read(path, String);
        content = replace(content, "\r\n" => "\n"); # Windows line endings become plain line breaks
        content = replace(content, '\ufeff' => "");  # drop the byte-order mark some editors add
    catch
        return ["responses.md could not be read"];
    end

    answers = Dict{Int,String}();
    issues = String[];

    # Separate each numbered prompt from the answer paragraphs that follow it.
    # A prompt starts with a number from 1 to 3, a period, and bold text (**...**) -
    for section in eachmatch(r"(?ms)^[ \t]{0,2}([1-3])\.[ \t]+\*\*.*?(?=^[ \t]{0,2}[1-3]\.[ \t]+\*\*|\z)", content)
        number = parse(Int, section.captures[1]);
        if !(number in questions)
            continue;
        end
        if haskey(answers, number)
            push!(issues, "Question $number appears more than once");
        end

        # The answer is everything after the first blank line; drop HTML comments -
        paragraphs = split(section.match, r"\n[ \t]*\n"; limit=2);
        answer = "";
        if length(paragraphs) == 2
            answer = paragraphs[2];
        end
        answers[number] = strip(replace(answer, r"(?s)<!--.*?-->" => ""));
    end

    # Report a missing answer or a remaining placeholder -
    for number in questions
        answer = get(answers, number, "");
        if !occursin(r"[\p{L}\p{N}]", answer)
            push!(issues, "Question $number has no answer");
        elseif occursin(r"(?i)\b(TODO|TBD)\b|Write your response|Your answer here", answer)
            push!(issues, "Question $number still has a placeholder");
        end
    end

    return issues;
end

println("PS4 submission check: Parts 1 and 2, 48 required tests, and questions 1–3.");
statuses = Dict{String,String}();
suites = Dict{String,Module}();
required_suites = ("testme_part_1.jl", "testme_part_2.jl");
# Run each test script in its own new module, so loading the student code twice does not replace it -
for filename in required_suites
    println("\nRunning ", filename);
    try
        suite = Module(gensym(:PS4Part));
        Base.include(suite, joinpath(@__DIR__, filename));
        suites[filename] = suite;
        statuses[filename] = "all tests passed";
    catch caught
        # A TestSetException means the tests ran and some failed; anything else stopped the script -
        if rootcause(caught) isa Test.TestSetException
            statuses[filename] = "some tests failed";
        else
            statuses[filename] = "tests could not run";
            showerror(stderr, caught, catch_backtrace());
            println(stderr);
        end
    end
end

issues = discussion_issues(RESPONSE_PATH);

# Both required test scripts must pass before the reports are made -
required_passed = true;
for name in required_suites
    if statuses[name] != "all tests passed"
        global required_passed = false; # global: change the variable outside this loop
    end
end

# Remove earlier reports so outputs/ holds only results from the code just checked -
# (instructor/validate.py checks that this list matches LogbookReports.REPORT_FILES)
for name in ("accuracy.csv", "confusion-matrix.csv", "confusion-matrix.png", "digit-weights.png",
        "misread-digits.png", "logbook-page.csv", "logbook-page.png", "logbook-summary.csv")
    rm(joinpath(CHECK_ROOT, "outputs", name); force=true);
end
report_status = "not generated because tests need attention";
if required_passed
    try
        ENV["GKSwstype"] = "100"; # render plots to files without opening a GUI window
        include(joinpath(@__DIR__, "src", "Reporting.jl"));
        println("\nTraining both digit readers and writing the reports.");
        assignment = getfield(suites["testme_part_2.jl"], :PlantLogbook);
        LogbookReports.write_outputs(assignment, joinpath(CHECK_ROOT, "outputs"));
        global report_status = "plots and data tables generated";
    catch caught
        global report_status = "OUTPUT GENERATION NEEDS ATTENTION";
        showerror(stderr, caught, catch_backtrace());
        println(stderr);
    end
end

ready = required_passed && report_status == "plots and data tables generated" && isempty(issues);

# Describe the written answers and the overall result, for the manifest and the summary -
if isempty(issues)
    discussion_status = "no unanswered prompts detected; instructor review required";
else
    discussion_status = join(issues, "; ");
end
if ready
    work_status = "checks passed; instructor review required";
else
    work_status = "WORK NEEDS ATTENTION";
end

# Record the test status and fingerprint every submitted source file -
files = [joinpath(@__DIR__, "Include.jl"), RESPONSE_PATH];
for (directory, _, filenames) in walkdir(joinpath(@__DIR__, "src"))
    for name in filenames
        file_path = joinpath(directory, name);
        # Fingerprint the implementation actually loaded, including in solution mode -
        if file_path == joinpath(@__DIR__, "src", "Compute.jl")
            file_path = COMPUTE_PATH;
        end
        push!(files, file_path);
    end
end
io = open(joinpath(CHECK_ROOT, "MANIFEST.txt"), "w");
if SOLUTION_MODE
    println(io, "PS4 instructor reference solution manifest");
else
    println(io, "PS4 CHEME 4800/5800 Fall 2026 submission manifest");
end
println(io, "Generated: ", Dates.now());
for name in sort(collect(keys(statuses)))
    println(io, name, ": ", statuses[name]);
end
println(io, "Outputs: ", report_status);
println(io, "Discussion: ", discussion_status);
println(io, "Required work: ", work_status);
for path in sort(files)
    relative = replace(relpath(path, @__DIR__), '\\' => '/');
    try
        stream = open(path);
        println(io, bytes2hex(sha256(stream)), "  ", relative); # the file's SHA-256 fingerprint
        close(stream);
    catch
        println(io, "MISSING OR UNREADABLE  ", relative);
    end
end
close(io);

println("\nSubmission summary");
for name in sort(collect(keys(statuses)))
    println(name, ": ", statuses[name]);
end
println("Outputs: ", report_status);
for issue in issues
    println("Discussion: ", issue);
end
if SOLUTION_MODE
    if ready
        println("Status: REFERENCE SOLUTION PASSED");
    else
        println("Status: REFERENCE SOLUTION NEEDS ATTENTION");
    end
    println("Wrote solution/MANIFEST.txt; reference outputs are in solution/outputs/.");
else
    if ready
        println("Status: READY TO PACKAGE; instructor review pending");
    else
        println("Status: WORK NEEDS ATTENTION");
    end
    println("Wrote MANIFEST.txt. Zip the assignment folder, including responses and outputs.");
    println("Name it CHEME-4800-5800-PS4-<your netid>.zip and upload it to Canvas yourself.");
    println("Submit your current work by the initial deadline even if some tests still fail.");
end
# Exit code 0 means every check passed; 1 means something needs attention -
if ready
    exit(0);
else
    exit(1);
end
