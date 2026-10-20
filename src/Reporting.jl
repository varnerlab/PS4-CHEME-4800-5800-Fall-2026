# ----------------------------------------------------------------------------
# Supplied reports | Train both readers and write the plots and tables.
# ----------------------------------------------------------------------------

"""Supplied plots and tables for the PS4 digit reader and the plant logbook page."""
module LogbookReports

using Plots  # confusion matrix, weight pictures, misread digits, and the logbook page
using Printf # tables with a fixed number of decimals

# Training settings used for every report -
const LOGISTIC_STEPS = 1000;   # gradient-descent steps per digit
const PERCEPTRON_PASSES = 20;  # passes through the training images per digit

# Every file write_outputs creates; the checker clears exactly these before a run -
const REPORT_FILES = ("accuracy.csv", "confusion-matrix.csv", "confusion-matrix.png",
    "digit-weights.png", "misread-digits.png", "logbook-page.csv", "logbook-page.png",
    "logbook-summary.csv");

const LOGBOOK_PATH = joinpath(@__DIR__, "..", "data", "logbook.csv");
const ENTRY_DIGITS = 4; # every logbook reading has four digits

"""
    write_outputs(assignment, directory) -> NamedTuple

Train both digit readers with the student's functions and write every report.

# Arguments

- `assignment`: the loaded `PlantLogbook` module, including the student's four
  completed functions.
- `directory`: the output folder; create it if needed.

# Returns and files

Return `(logistic, perceptron, digit_accuracy, entry_accuracy)`, where the first
two are the testing accuracies and the last two describe the logbook entries read by
logistic regression. Write or overwrite the eight files in `REPORT_FILES`. Print
progress, both accuracies, and a text copy of the logbook page.

# Errors

Throw `ErrorException` if a student function returns results of the wrong size.
Errors from the student functions, file writing, or plotting propagate.
"""
function write_outputs(assignment, directory)

    # Call each student function once on a tiny input, so an unfinished one stops here -
    assignment.digit_labels([0, 1], 0);
    assignment.train_digit_models([1.0 1.0; 0.5 1.0], [0, 1]; maxiter=1);
    assignment.predict_digits(zeros(2, 10), zeros(1, 2));
    assignment.confusion_matrix([0], [0]);

    # Train both readers on the training images -
    mkpath(directory);
    data = assignment.load_digits();
    training, testing = data.training, data.testing;
    W = Dict{Symbol,Array{Float64,2}}();
    for (method, name, steps, unit) in ((:logistic, "logistic regression", LOGISTIC_STEPS, "steps"),
                                        (:perceptron, "the Perceptron", PERCEPTRON_PASSES, "passes"))
        println("Training $name ($steps $unit for each of the 10 digits) on ",
            size(training.X, 1), " images. This can take a minute.");
        seconds = @elapsed W[method] = assignment.train_digit_models(training.X, training.labels;
            method=method, maxiter=steps);
        size(W[method]) == (size(training.X, 2), 10) ||
            error("train_digit_models must return a $(size(training.X, 2)) × 10 matrix, not $(size(W[method]))");
        @printf("  done in %.1f s\n", seconds);
    end

    # Read the testing images with both readers -
    predicted = Dict(method => assignment.predict_digits(W[method], testing.X) for method in keys(W));
    for method in keys(W)
        length(predicted[method]) == length(testing.labels) ||
            error("predict_digits must return one digit per testing image");
    end
    CM = Dict(method => assignment.confusion_matrix(testing.labels, predicted[method]) for method in keys(W));
    for method in keys(W)
        size(CM[method]) == (10, 10) && sum(CM[method]) == length(testing.labels) ||
            error("confusion_matrix must return a 10 × 10 matrix that counts every testing image");
    end

    # Write the reports -
    write_accuracy(directory, testing.labels, predicted);
    write_confusion(directory, CM[:logistic]);
    write_weights(directory, W[:logistic]);
    write_misread(directory, testing, predicted[:logistic], CM[:logistic]);
    page = write_logbook(directory, testing, predicted[:logistic]);
    summary = write_summary(directory, testing.labels, predicted[:logistic]);

    # Print what the reports show -
    accuracy(method) = count(predicted[method] .== testing.labels) / length(testing.labels);
    println();
    @printf("Testing accuracy, logistic regression: %.4f (%d of %d digits)\n",
        accuracy(:logistic), count(predicted[:logistic] .== testing.labels), length(testing.labels));
    @printf("Testing accuracy, Perceptron:          %.4f (%d of %d digits)\n",
        accuracy(:perceptron), count(predicted[:perceptron] .== testing.labels), length(testing.labels));
    @printf("Four-digit logbook entries read correctly: %.4f (%d of %d entries)\n",
        summary.entry_accuracy, summary.entries_correct, summary.entries);
    println("\nLogbook page, read by logistic regression:");
    for row in page
        @printf("  %s  %-20s  true %6s %-3s  read %6s %-3s%s\n", row.time, row.quantity,
            row.value, row.unit, row.read, row.unit, row.match ? "" : "  <- misread");
    end
    println("\nWrote the plots and tables to ", abspath(directory));

    return (logistic = accuracy(:logistic), perceptron = accuracy(:perceptron),
        digit_accuracy = summary.digit_accuracy, entry_accuracy = summary.entry_accuracy);
end


# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------

"""Write accuracy.csv: correct readings for each method, overall and for each digit."""
function write_accuracy(directory, labels, predicted)

    open(joinpath(directory, "accuracy.csv"), "w") do io
        println(io, "method,digit,correct,total,accuracy");
        for method in (:logistic, :perceptron)
            correct = predicted[method] .== labels;
            @printf(io, "%s,all,%d,%d,%.4f\n", method, count(correct), length(labels), count(correct) / length(labels));
            for d in 0:9
                mask = labels .== d;
                @printf(io, "%s,%d,%d,%d,%.4f\n", method, d, count(correct .& mask), count(mask),
                    count(correct .& mask) / count(mask));
            end
        end
    end
end


"""Write logbook-summary.csv: digit accuracy and four-digit entry accuracy on the testing set."""
function write_summary(directory, labels, predicted)

    correct = predicted .== labels;
    entries = length(labels) ÷ ENTRY_DIGITS; # consecutive testing images, four at a time
    entries_correct = count(all(correct[(ENTRY_DIGITS * (k - 1) + 1):(ENTRY_DIGITS * k)]) for k in 1:entries);
    summary = (digits = length(labels), digits_correct = count(correct),
        digit_accuracy = count(correct) / length(labels), entries = entries,
        entries_correct = entries_correct, entry_accuracy = entries_correct / entries);

    open(joinpath(directory, "logbook-summary.csv"), "w") do io
        println(io, "measure,value");
        println(io, "digits_read,", summary.digits);
        println(io, "digits_correct,", summary.digits_correct);
        @printf(io, "digit_accuracy,%.4f\n", summary.digit_accuracy);
        println(io, "entries_read,", summary.entries);
        println(io, "entries_correct,", summary.entries_correct);
        @printf(io, "entry_accuracy,%.4f\n", summary.entry_accuracy);
    end

    return summary;
end


# ----------------------------------------------------------------------------
# Pictures
# ----------------------------------------------------------------------------

"""Return image `i` of `X` as a 28 × 28 picture, top row first."""
picture(X, i) = permutedims(reshape(X[i, 1:784], 28, 28));

const INK = cgrad([:white, :black]); # 0 is white paper, 1 is black ink

"""Draw one picture with no axes; `title` goes above it."""
function draw(img; title="", titlecolor=:black, clims=(0, 1), color=INK)
    return heatmap(img; yflip=true, color=color, clims=clims, aspect_ratio=:equal,
        axis=false, ticks=false, colorbar=false, framestyle=:none, title=title,
        titlefontsize=9, titlefontcolor=titlecolor);
end


"""Write confusion-matrix.csv and confusion-matrix.png for logistic regression."""
function write_confusion(directory, CM)

    open(joinpath(directory, "confusion-matrix.csv"), "w") do io
        println(io, "true_digit,", join(("read_as_$d" for d in 0:9), ","));
        for a in 0:9
            println(io, a, ",", join(CM[a + 1, :], ","));
        end
    end

    # Shade by count, capped at the largest mix-up so the mix-ups stand out -
    largest = maximum(CM[a, p] for a in 1:10, p in 1:10 if a != p);
    theme(:default);
    p = heatmap(0:9, 0:9, min.(CM, largest); yflip=true, color=:Blues, clims=(0, largest),
        aspect_ratio=:equal, xticks=0:9, yticks=0:9, xlabel="Digit the model read",
        ylabel="True digit", title="Confusion matrix (logistic regression, testing images)",
        titlefontsize=11, colorbar_title="count (mix-ups)", size=(720, 620));
    for a in 0:9, r in 0:9
        n = CM[a + 1, r + 1];
        annotate!(p, r, a, text(string(n), 8, n >= 0.6 * largest ? :white : :black));
    end
    savefig(p, joinpath(directory, "confusion-matrix.png"));
end


"""Write digit-weights.png: each digit's 784 pixel weights drawn as a 28 × 28 picture."""
function write_weights(directory, W)

    panels = map(0:9) do d
        img = permutedims(reshape(W[1:784, d + 1], 28, 28));
        limit = maximum(abs, img);
        draw(img; title="Weights for digit $d", color=:RdBu, clims=(-limit, limit));
    end
    theme(:default);
    p = plot(panels...; layout=(2, 5), size=(1000, 460),
        plot_title="Logistic regression weights (blue: ink here raises the score; red: lowers it)",
        plot_titlefontsize=11);
    savefig(p, joinpath(directory, "digit-weights.png"));
end


"""
    write_misread(directory, testing, predicted, CM)

Write misread-digits.png: one row for each of the six most common mix-ups in the
confusion matrix `CM` (largest count first; ties in true-digit, then read-digit order),
showing the first four testing images with that mix-up.
"""
function write_misread(directory, testing, predicted, CM)

    mixups = sort([(CM[a + 1, r + 1], a, r) for a in 0:9 for r in 0:9 if a != r];
        by = m -> (-m[1], m[2], m[3]))[1:6];
    panels = Any[];
    for (n, a, r) in mixups
        shown = findall((testing.labels .== a) .& (predicted .== r))[1:min(4, n)];
        append!(panels, [draw(picture(testing.X, i); title="true $a, read $r") for i in shown]);
        append!(panels, [plot(; framestyle=:none) for _ in (length(shown) + 1):4]); # pad a short row
    end
    theme(:default);
    p = plot(panels...; layout=(6, 4), size=(620, 960),
        plot_title="Six most common mix-ups (logistic regression), four examples each",
        plot_titlefontsize=11);
    savefig(p, joinpath(directory, "misread-digits.png"));
end


"""
    write_logbook(directory, testing, predicted) -> Vector{NamedTuple}

Write logbook-page.csv and logbook-page.png from data/logbook.csv. Each entry's
digits are read from the listed testing images; a decimal point is copied, not read.
Return one `(time, quantity, unit, value, read, match)` row per entry.
"""
function write_logbook(directory, testing, predicted)

    # Read the page layout and replace each digit with the digit the model read -
    lines = readlines(LOGBOOK_PATH)[2:end];
    page = map(lines) do line
        time, quantity, unit, value, images... = split(line, ",");
        numbers = parse.(Int, images);
        digits = string.(predicted[numbers]);
        k = 0;
        read = join(c == '.' ? "." : digits[k += 1] for c in value);
        (time = String(time), quantity = String(quantity), unit = String(unit),
            value = String(value), read = read, match = read == value, images = numbers);
    end

    open(joinpath(directory, "logbook-page.csv"), "w") do io
        println(io, "time,quantity,unit,true_value,read_value,match");
        for row in page
            println(io, join((row.time, row.quantity, row.unit, row.value, row.read, row.match), ","));
        end
    end

    # Draw each entry as its four handwritten digits side by side -
    panels = map(page) do row
        img = reduce(hcat, picture(testing.X, i) for i in row.images);
        label = "$(row.time) $(row.quantity) ($(row.unit))\ntrue $(row.value), read $(row.read)";
        draw(img; title=row.match ? label : label * "  MISREAD", titlecolor=row.match ? :black : :red);
    end
    theme(:default);
    p = plot(panels...; layout=(5, 4), size=(1300, 900),
        plot_title="Plant logbook page, read by logistic regression (red: at least one digit misread)",
        plot_titlefontsize=12);
    savefig(p, joinpath(directory, "logbook-page.png"));

    return page;
end

end
