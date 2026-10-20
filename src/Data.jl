# ----------------------------------------------------------------------------
# Supplied loader | Read the handwritten-digit images and labels from data/.
# ----------------------------------------------------------------------------

const PIXELS = 28 * 28; # pixels in one image
const DATA_FILES = ("train-images.bin", "train-labels.bin", "test-images.bin", "test-labels.bin");

"""
    load_digits(directory::AbstractString = <the assignment's data folder>) -> NamedTuple

Read the training and test images and their digit labels.

# Returns

A named tuple `(training, test)`. Each part is a named tuple `(X, labels)`:

- `X::Matrix{Float64}`: one row per image and 785 columns. Columns 1 to 784 hold the
  pixels, each between 0 (white paper) and 1 (black ink). The last column is all
  ones: it is the 1 for the bias, as in the L9 labs. The training part has 10,000
  rows and the test part has 10,000 rows.
- `labels::Vector{Int}`: the true digit (0 to 9) in each image, in the same order
  as the rows of `X`.

The pixels of an image are stored row by row: columns 1 to 28 are the top row of
the picture, columns 29 to 56 the next row, and so on. To turn row `i` of `X` back
into a 28 × 28 picture, use `permutedims(reshape(X[i, 1:784], 28, 28))`.

# Errors

Throws an `ErrorException` that names the file if a data file is missing or has
the wrong size, which usually means the download or unzip was incomplete.
"""
function load_digits(directory::AbstractString = joinpath(@__DIR__, "..", "data"))

    # Check that every file is present and complete before reading any of them -
    expected = Dict("train-images.bin" => 10_000 * PIXELS, "train-labels.bin" => 10_000,
                    "test-images.bin" => 10_000 * PIXELS, "test-labels.bin" => 10_000);
    for name in DATA_FILES
        path = joinpath(directory, name);
        isfile(path) || error("data file $name is missing from $directory. Download the assignment again.");
        n = filesize(path);
        n == expected[name] || error("data file $name has $n bytes; expected $(expected[name]). Download the assignment again.");
    end

    # Read one image set: 784 bytes per image become one row of X, plus the bias -
    function read_set(prefix)
        labels = Int.(read(joinpath(directory, "$prefix-labels.bin")));
        count = length(labels);
        bytes = Array{UInt8}(undef, PIXELS, count); # column j holds image j
        read!(joinpath(directory, "$prefix-images.bin"), bytes);
        X = [permutedims(Float64.(bytes)) ./ 255 ones(count)];
        return (X = X, labels = labels);
    end

    return (training = read_set("train"), test = read_set("test"));
end
