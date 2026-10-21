# ----------------------------------------------------------------------------
# Supplied loader | Read the handwritten-digit images and labels from data/.
# ----------------------------------------------------------------------------

const PIXELS = 28 * 28; # pixels in one image
const DATA_FILES = ("train-images.bin", "train-labels.bin", "test-images.bin", "test-labels.bin");

"""
    load_digits(directory::String = <the assignment's data folder>) -> NamedTuple

Read the training and testing images and their digit labels.

### Returns
A named tuple `(training, testing)`, as in the labs. Each part is a named tuple
`(X, labels)`:
- `X::Array{Float64,2}`: the augmented data matrix, one row per image and 785
  columns. Columns 1 to 784 hold the pixels, each between 0 (white paper) and 1
  (black ink). The last column is all ones: it is the 1 for the bias, as in the L9
  labs. The training part has 10,000 rows and the testing part has 10,000 rows.
- `labels::Array{Int64,1}`: the true digit (0 to 9) in each image, in the same order
  as the rows of `X`.

The pixels of an image are stored row by row: columns 1 to 28 are the top row of
the picture, columns 29 to 56 the next row, and so on. To turn row `i` of `X` back
into a 28 × 28 picture, use `permutedims(reshape(X[i, 1:784], 28, 28))`.

### Errors
Throws an `ErrorException` that names the file if a data file is missing or has
the wrong size, which usually means the download or unzip was incomplete.
"""
function load_digits(directory::String = joinpath(@__DIR__, "..", "data"))

    # Check that every file is present and complete before reading any of them -
    expected = Dict("train-images.bin" => 10_000 * PIXELS, "train-labels.bin" => 10_000,
                    "test-images.bin" => 10_000 * PIXELS, "test-labels.bin" => 10_000);
    for name in DATA_FILES
        path = joinpath(directory, name);
        if !isfile(path)
            error("data file $name is missing from $directory. Download the assignment again.");
        end
        n = filesize(path); # the file size in bytes
        if n != expected[name]
            error("data file $name has $n bytes; expected $(expected[name]). Download the assignment again.");
        end
    end

    # Read both image sets -
    training = read_image_set(directory, "train");
    testing = read_image_set(directory, "test");

    return (training = training, testing = testing);
end


"""
    read_image_set(directory::String, prefix::String) -> NamedTuple

Read one image set, `prefix` = `"train"` or `"test"`, from the files
`<prefix>-images.bin` and `<prefix>-labels.bin` in `directory`. Return `(X, labels)`
as described in `load_digits`. `load_digits` checks the files before calling this.
"""
function read_image_set(directory::String, prefix::String)

    # Read the labels: one byte per image, holding its digit -
    label_bytes = read(joinpath(directory, "$prefix-labels.bin"));
    labels = Int64.(label_bytes); # turn each byte into an integer
    number_of_images = length(labels);

    # Read the pixels: 784 bytes per image, from 0 (white paper) to 255 (black ink) -
    bytes = Array{UInt8}(undef, PIXELS, number_of_images); # column i will hold image i
    read!(joinpath(directory, "$prefix-images.bin"), bytes);

    # Build the augmented data matrix: one row per image, the pixels from 0 to 1, then a 1 -
    X = ones(number_of_images, PIXELS + 1); # the last column stays 1: the 1 for the bias
    for i in 1:number_of_images
        X[i, 1:PIXELS] = bytes[:, i] ./ 255; # image i, scaled so that black ink is 1
    end

    return (X = X, labels = labels);
end
