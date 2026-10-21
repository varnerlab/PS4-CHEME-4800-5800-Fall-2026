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

    # Load the images -
    mkpath(directory);
    data = assignment.load_digits();
    training = data.training;
    testing = data.testing;
    number_of_rows = size(training.X, 2);           # rows of W: one per column of X
    number_of_images = length(testing.labels);      # testing images to read

    # Train logistic regression on the training images -
    println("Training logistic regression ($LOGISTIC_STEPS steps for each of the 10 digits) on ",
        size(training.X, 1), " images.");
    W_logistic = assignment.train_digit_models(training.X, training.labels;
        method=:logistic, maxiter=LOGISTIC_STEPS);
    check_parameters(W_logistic, number_of_rows);
    println("  done");

    # Train the Perceptron on the training images -
    println("Training the Perceptron ($PERCEPTRON_PASSES passes for each of the 10 digits) on ",
        size(training.X, 1), " images.");
    W_perceptron = assignment.train_digit_models(training.X, training.labels;
        method=:perceptron, maxiter=PERCEPTRON_PASSES);
    check_parameters(W_perceptron, number_of_rows);
    println("  done");

    # Read the testing images with both readers -
    predicted_logistic = assignment.predict_digits(W_logistic, testing.X);
    predicted_perceptron = assignment.predict_digits(W_perceptron, testing.X);
    if length(predicted_logistic) != number_of_images || length(predicted_perceptron) != number_of_images
        error("predict_digits must return one digit per testing image");
    end

    # Count the mix-ups of both readers -
    CM_logistic = assignment.confusion_matrix(testing.labels, predicted_logistic);
    CM_perceptron = assignment.confusion_matrix(testing.labels, predicted_perceptron);
    check_confusion(CM_logistic, number_of_images);
    check_confusion(CM_perceptron, number_of_images);

    # Write the reports -
    write_accuracy(directory, testing.labels, predicted_logistic, predicted_perceptron);
    write_confusion(directory, CM_logistic);
    write_weights(directory, W_logistic);
    write_misread(directory, testing, predicted_logistic, CM_logistic);
    page = write_logbook(directory, testing, predicted_logistic);
    summary = write_summary(directory, testing.labels, predicted_logistic);

    # Compute both testing accuracies -
    correct_logistic = count(predicted_logistic .== testing.labels);
    correct_perceptron = count(predicted_perceptron .== testing.labels);
    accuracy_logistic = correct_logistic / number_of_images;
    accuracy_perceptron = correct_perceptron / number_of_images;

    # Print what the reports show -
    println();
    @printf("Testing accuracy, logistic regression: %.4f (%d of %d digits)\n",
        accuracy_logistic, correct_logistic, number_of_images);
    @printf("Testing accuracy, Perceptron:          %.4f (%d of %d digits)\n",
        accuracy_perceptron, correct_perceptron, number_of_images);
    @printf("Four-digit logbook entries read correctly: %.4f (%d of %d entries)\n",
        summary.entry_accuracy, summary.entries_correct, summary.entries);
    println("\nLogbook page, read by logistic regression:");
    for row in page
        if row.match
            note = "";
        else
            note = "  <- misread";
        end
        @printf("  %s  %-20s  true %6s %-3s  read %6s %-3s%s\n", row.time, row.quantity,
            row.value, row.unit, row.read, row.unit, note);
    end
    println("\nWrote the plots and tables to ", abspath(directory));

    return (logistic = accuracy_logistic, perceptron = accuracy_perceptron,
        digit_accuracy = summary.digit_accuracy, entry_accuracy = summary.entry_accuracy);
end


# ----------------------------------------------------------------------------
# Checks on the student results
# ----------------------------------------------------------------------------

"""
    check_parameters(W, number_of_rows)

Throw an `ErrorException` unless `W`, returned by `train_digit_models`, has
`number_of_rows` rows and 10 columns.
"""
function check_parameters(W, number_of_rows)

    if size(W) != (number_of_rows, 10)
        error("train_digit_models must return a $number_of_rows × 10 matrix, not $(size(W))");
    end
end


"""
    check_confusion(CM, number_of_images)

Throw an `ErrorException` unless `CM`, returned by `confusion_matrix`, is 10 × 10
and counts all `number_of_images` testing images.
"""
function check_confusion(CM, number_of_images)

    if size(CM) != (10, 10) || sum(CM) != number_of_images
        error("confusion_matrix must return a 10 × 10 matrix that counts every testing image");
    end
end


# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------

"""
    write_accuracy(directory, labels, predicted_logistic, predicted_perceptron)

Write accuracy.csv: the correct readings of each method, for all testing images and
for each digit.
"""
function write_accuracy(directory, labels, predicted_logistic, predicted_perceptron)

    io = open(joinpath(directory, "accuracy.csv"), "w");
    println(io, "method,digit,correct,total,accuracy");
    write_accuracy_rows(io, "logistic", labels, predicted_logistic);
    write_accuracy_rows(io, "perceptron", labels, predicted_perceptron);
    close(io);
end


"""
    write_accuracy_rows(io, method, labels, predicted)

Write the accuracy.csv rows for one method to the open file `io`: one row for all
testing images, then one row for each digit from 0 to 9.
"""
function write_accuracy_rows(io, method, labels, predicted)

    # All testing images -
    correct = count(predicted .== labels);
    total = length(labels);
    @printf(io, "%s,all,%d,%d,%.4f\n", method, correct, total, correct / total);

    # Each digit: count its images, and how many of them were read correctly -
    for d in 0:9
        total = 0;   # images of digit d
        correct = 0; # images of digit d read as d
        for i in 1:length(labels)
            if labels[i] == d
                total += 1;
                if predicted[i] == d
                    correct += 1;
                end
            end
        end
        @printf(io, "%s,%d,%d,%d,%.4f\n", method, d, correct, total, correct / total);
    end
end


"""
    write_summary(directory, labels, predicted) -> NamedTuple

Write logbook-summary.csv: digit accuracy and four-digit entry accuracy on the
testing set. Each entry is four consecutive testing images, and it counts as correct
only when all four digits are read correctly. Return the numbers as a named tuple.
"""
function write_summary(directory, labels, predicted)

    # Single digits -
    digits = length(labels);
    digits_correct = count(predicted .== labels);

    # Four-digit entries: images 1 to 4 are entry 1, images 5 to 8 are entry 2, and so on -
    entries = div(length(labels), ENTRY_DIGITS); # whole entries only
    entries_correct = 0;
    for k in 1:entries
        first_image = ENTRY_DIGITS * (k - 1) + 1; # the first image of entry k
        last_image = ENTRY_DIGITS * k;            # the last image of entry k
        if predicted[first_image:last_image] == labels[first_image:last_image]
            entries_correct += 1;
        end
    end

    summary = (digits = digits, digits_correct = digits_correct,
        digit_accuracy = digits_correct / digits, entries = entries,
        entries_correct = entries_correct, entry_accuracy = entries_correct / entries);

    # Write the table -
    io = open(joinpath(directory, "logbook-summary.csv"), "w");
    println(io, "measure,value");
    println(io, "digits_read,", summary.digits);
    println(io, "digits_correct,", summary.digits_correct);
    @printf(io, "digit_accuracy,%.4f\n", summary.digit_accuracy);
    println(io, "entries_read,", summary.entries);
    println(io, "entries_correct,", summary.entries_correct);
    @printf(io, "entry_accuracy,%.4f\n", summary.entry_accuracy);
    close(io);

    return summary;
end


# ----------------------------------------------------------------------------
# Pictures
# ----------------------------------------------------------------------------

const INK = cgrad([:white, :black]); # 0 is white paper, 1 is black ink

"""
    picture(X, i) -> Array{Float64,2}

Return image `i`, row `i` of `X`, as a 28 × 28 picture, top row first.
"""
function picture(X, i)

    pixels = X[i, 1:784]; # drop the 1 for the bias
    return permutedims(reshape(pixels, 28, 28));
end


"""
    draw(img; title="", titlecolor=:black, clims=(0, 1), color=INK)

Draw one picture with no axes; `title` goes above it. `clims` sets the values shown
as the two ends of the color scale `color`.
"""
function draw(img; title="", titlecolor=:black, clims=(0, 1), color=INK)

    return heatmap(img; yflip=true, color=color, clims=clims, aspect_ratio=:equal,
        axis=false, ticks=false, colorbar=false, framestyle=:none, title=title,
        titlefontsize=9, titlefontcolor=titlecolor);
end


"""
    write_confusion(directory, CM)

Write confusion-matrix.csv and confusion-matrix.png for logistic regression.
"""
function write_confusion(directory, CM)

    # Write the table: one row per true digit, one column per digit read -
    io = open(joinpath(directory, "confusion-matrix.csv"), "w");
    header = "true_digit";
    for d in 0:9
        header = header * ",read_as_$d";
    end
    println(io, header);
    for a in 0:9
        println(io, a, ",", join(CM[a + 1, :], ","));
    end
    close(io);

    # Find the largest mix-up: the largest count off the diagonal -
    largest = 0;
    for a in 1:10
        for p in 1:10
            if a != p && CM[a, p] > largest
                largest = CM[a, p];
            end
        end
    end

    # Shade by count, capped at the largest mix-up so the mix-ups stand out -
    shade = min.(CM, largest); # every count above largest is shaded as largest
    theme(:default);
    figure = heatmap(0:9, 0:9, shade; yflip=true, color=:Blues, clims=(0, largest),
        aspect_ratio=:equal, xticks=0:9, yticks=0:9, xlabel="Digit the model read",
        ylabel="True digit", title="Confusion matrix (logistic regression, testing images)",
        titlefontsize=11, colorbar_title="count (mix-ups)", size=(720, 620));

    # Write each count on its cell: white text on dark cells, black text on light cells -
    for a in 0:9
        for r in 0:9
            n = CM[a + 1, r + 1];
            if n >= 0.6 * largest
                text_color = :white;
            else
                text_color = :black;
            end
            annotate!(figure, r, a, text(string(n), 8, text_color));
        end
    end
    savefig(figure, joinpath(directory, "confusion-matrix.png"));
end


"""
    write_weights(directory, W)

Write digit-weights.png: each digit's 784 pixel weights drawn as a 28 × 28 picture.
"""
function write_weights(directory, W)

    # Draw one panel per digit -
    panels = [];
    for d in 0:9
        weights = W[1:784, d + 1]; # column d + 1 holds digit d; drop the bias
        img = permutedims(reshape(weights, 28, 28));
        limit = maximum(abs.(img)); # the largest weight, positive or negative, sets the color scale
        push!(panels, draw(img; title="Weights for digit $d", color=:RdBu, clims=(-limit, limit)));
    end

    # Put the ten panels in two rows of five; panels... passes each panel to plot -
    theme(:default);
    figure = plot(panels...; layout=(2, 5), size=(1000, 460),
        plot_title="Logistic regression weights (blue: ink here raises the score; red: lowers it)",
        plot_titlefontsize=11);
    savefig(figure, joinpath(directory, "digit-weights.png"));
end


"""
    write_misread(directory, testing, predicted, CM)

Write misread-digits.png: one row for each of the six most common mix-ups in the
confusion matrix `CM` (largest count first; ties in true-digit, then read-digit order),
showing the first four testing images with that mix-up.
"""
function write_misread(directory, testing, predicted, CM)

    # Find the six most common mix-ups. Copy the counts and mark the diagonal (the
    # correct readings) with -1, then take the largest count six times. The search
    # goes row by row, so a tie goes to the smaller true digit, then the smaller digit read -
    counts = copy(CM);
    for k in 1:10
        counts[k, k] = -1;
    end
    mixup_true = zeros(Int, 6); # the true digit of each mix-up, most common first
    mixup_read = zeros(Int, 6); # the digit it was read as
    for m in 1:6
        best = -1;
        for a in 0:9
            for r in 0:9
                if counts[a + 1, r + 1] > best
                    best = counts[a + 1, r + 1];
                    mixup_true[m] = a;
                    mixup_read[m] = r;
                end
            end
        end
        counts[mixup_true[m] + 1, mixup_read[m] + 1] = -1; # do not pick this mix-up again
    end

    # Draw the first four testing images of each mix-up, one row per mix-up -
    panels = [];
    for m in 1:6
        a = mixup_true[m];
        r = mixup_read[m];
        shown = 0; # images drawn in this row so far
        for i in 1:length(testing.labels)
            if shown < 4 && testing.labels[i] == a && predicted[i] == r
                push!(panels, draw(picture(testing.X, i); title="true $a, read $r"));
                shown += 1;
            end
        end

        # Fill a short row with blank panels, so every row has four -
        while shown < 4
            push!(panels, plot(framestyle=:none));
            shown += 1;
        end
    end

    theme(:default);
    figure = plot(panels...; layout=(6, 4), size=(620, 960),
        plot_title="Six most common mix-ups (logistic regression), four examples each",
        plot_titlefontsize=11);
    savefig(figure, joinpath(directory, "misread-digits.png"));
end


"""
    write_logbook(directory, testing, predicted) -> Array{Any,1}

Write logbook-page.csv and logbook-page.png from data/logbook.csv. Each entry's
digits are read from the listed testing images; a decimal point is copied, not read.
Return one `(time, quantity, unit, value, read, match, images)` row per entry.
"""
function write_logbook(directory, testing, predicted)

    # Read each entry of the page, skipping the header line -
    lines = readlines(LOGBOOK_PATH);
    page = [];
    for line in lines[2:end]
        fields = split(line, ",");
        time = String(fields[1]);
        quantity = String(fields[2]);
        unit = String(fields[3]);
        value = String(fields[4]);           # the true reading, such as "342.5"
        images = parse.(Int, fields[5:end]); # the testing images that hold its digits

        # Build the reading: copy each decimal point, and replace each digit with the
        # digit the model read from the next image -
        reading = "";
        k = 0; # images used so far
        for c in value
            if c == '.'
                reading = reading * ".";
            else
                k += 1;
                reading = reading * string(predicted[images[k]]);
            end
        end

        push!(page, (time = time, quantity = quantity, unit = unit, value = value,
            read = reading, match = (reading == value), images = images));
    end

    # Write the table -
    io = open(joinpath(directory, "logbook-page.csv"), "w");
    println(io, "time,quantity,unit,true_value,read_value,match");
    for row in page
        println(io, join((row.time, row.quantity, row.unit, row.value, row.read, row.match), ","));
    end
    close(io);

    # Draw each entry as its four handwritten digits side by side -
    panels = [];
    for row in page
        img = picture(testing.X, row.images[1]);
        for k in 2:length(row.images)
            img = hcat(img, picture(testing.X, row.images[k])); # add the next digit on the right
        end
        label = "$(row.time) $(row.quantity) ($(row.unit))\ntrue $(row.value), read $(row.read)";
        if row.match
            push!(panels, draw(img; title=label, titlecolor=:black));
        else
            push!(panels, draw(img; title=label * "  MISREAD", titlecolor=:red));
        end
    end

    theme(:default);
    figure = plot(panels...; layout=(5, 4), size=(1300, 900),
        plot_title="Plant logbook page, read by logistic regression (red: at least one digit misread)",
        plot_titlefontsize=12);
    savefig(figure, joinpath(directory, "logbook-page.png"));

    return page;
end

end
