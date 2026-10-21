# ----------------------------------------------------------------------------
# Training functions | The binary classifiers from the L9b and L9d labs.
# ----------------------------------------------------------------------------
# These are the completed lab functions, with the same names, arguments, and
# results. Each one trains a single binary classifier with labels y = ±1: y = 1
# for one class and y = -1 for the other. Two changes make them fast enough for 10,000 images; neither changes
# the math. my_perceptron reads the images from a transposed copy of X, and
# my_logistic_regression computes the gradient for all images at once.

"""
    my_perceptron(X::Array{Float64,2}, y::Array{Float64,1};
        maxiter::Int64 = 1000, mistakes::Int64 = 0) -> NamedTuple

Learn the parameters `θ` of a linear classifier with the online Perceptron algorithm
from the L9a lecture. This is the L9b lab function.

The classifier gives example `i` the label of the sign of its score, `x̂ᵢᵀθ`. Training
starts from `θ = 0` and passes through the examples in row order, one at a time.
Example `i` is a mistake when `yᵢ x̂ᵢᵀθ ≤ 0`, that is, when its score has the wrong
sign or is zero; after each mistake, the parameters are updated before the next
example is visited. Training stops after the first pass with at most `mistakes`
mistakes, or after `maxiter` passes, whichever comes first.

On data that are not linearly separable, every pass makes at least one mistake, so
with `mistakes = 0` the loop runs all `maxiter` passes.

### Arguments
- `X::Array{Float64,2}`: the augmented features, one row per example: the features
  (in PS4, the 784 pixel values of an image), followed by a 1 for the bias. For `n`
  examples with `m` features, `X` is `n × (m+1)`.
- `y::Array{Float64,1}`: the label of each example, `1` or `-1`, in the same order as
  the rows of `X`.
- `maxiter::Int64 = 1000`: the maximum number of passes through the training data,
  `T` in L9a. At least 1.
- `mistakes::Int64 = 0`: the mistake threshold, `M` in L9a. Training stops after the
  first pass with at most this many mistakes.

### Returns
A `NamedTuple` with the fields:
- `θ::Array{Float64,1}`: the learned parameters, one per column of `X`: the feature
  weights, followed by the bias.
- `passes::Int64`: the number of passes made, at most `maxiter`.
- `mistakes::Int64`: the number of mistakes in the last pass.

### Errors
Throws an `ArgumentError` if `X` has no rows, a label is not `1` or `-1`, or
`maxiter < 1`, and a `DimensionMismatch` if `y` does not have one entry per row of `X`.

### Example
```julia
y = digit_labels(training.labels, 3); # 1 for each image of a 3, -1 for the rest
result = my_perceptron(training.X, y, maxiter = 20);
result.θ, result.passes, result.mistakes
```
"""
function my_perceptron(X::Array{Float64,2}, y::Array{Float64,1};
    maxiter::Int64 = 1000, mistakes::Int64 = 0)::NamedTuple

    # Check the arguments -
    number_of_examples = size(X,1); # the number of training examples, one per row of X
    if number_of_examples < 1
        throw(ArgumentError("X must have at least one row"))
    end
    if length(y) != number_of_examples
        throw(DimensionMismatch("y must have one entry per row of X"))
    end
    if !all((y .== 1) .| (y .== -1)) # .| is "or", applied entry by entry
        throw(ArgumentError("every label in y must be 1 or -1"))
    end
    if maxiter < 1
        throw(ArgumentError("maxiter must be at least 1"))
    end

    # Initialize, with the names from L9a -
    T = maxiter; # the maximum number of passes through the training data
    M = mistakes; # the mistake threshold: stop after a pass with at most M mistakes
    θ = zeros(size(X,2)); # the parameters: the feature weights, then the bias
    Xᵀ = permutedims(X); # column i of Xᵀ is row i of X; reading columns is faster

    # Main loop: one pass through the training data for each value of t -
    for t ∈ 1:T

        # Step 1: start the pass with no mistakes -
        number_of_mistakes = 0; # the number of mistakes in this pass so far

        # Step 2: visit the training examples in order, one at a time -
        for i ∈ 1:number_of_examples
            x̂ = view(Xᵀ, :, i); # the augmented features of example i: row i of X

            # The mistake test, yᵢ x̂ᵢᵀθ ≤ 0 -
            if y[i]*dot(x̂, θ) ≤ 0

                # The update, θ ← θ + yᵢ x̂ᵢ -
                θ .+= y[i] .* x̂;
                number_of_mistakes += 1; # count the mistake
            end
        end

        # The stopping test, step 4 of L9a: at most M mistakes, or t has reached T -
        if (number_of_mistakes ≤ M || t ≥ T)
            return (θ = θ, passes = t, mistakes = number_of_mistakes);
        end
    end
end


"""
    my_logistic_regression(X::Array{<:Number,2}, y::Array{<:Number,1}; α::Float64,
        β::Float64 = 1.0, δ::Float64 = 0.0, ϵ::Float64 = 1e-6,
        maxiter::Int64 = 50_000) -> NamedTuple

Estimate the parameters `θ` of a logistic regression classifier by minimizing the
regularized cross-entropy loss from the L9c lecture with gradient descent. This is
the L9d lab function.

The loss is `J_δ(θ) = Σᵢ log(1 + exp(-uᵢ)) + (δ/2)‖θ‖²`, where `uᵢ = 2β yᵢ x̂ᵢᵀθ`, and its
gradient is `∇J_δ(θ) = -2β Σᵢ (1 - σ(uᵢ)) yᵢ x̂ᵢ + δθ`, where `σ` is the logistic
function. Training starts from `θ = 0` and takes the step `θ ← θ - α∇J_δ(θ)` with a
constant learning rate `α`. It stops when a step changes `θ` by at most `ϵ`, measured
with the Euclidean norm, or after `maxiter` steps, whichever comes first.

A learning rate `α ≤ 1/(L + δ)`, where `L = β²‖X‖₂²` and `‖X‖₂` is the largest singular
value of `X`, guarantees that no step increases the loss. In Julia, `‖X‖₂` is
`opnorm(X)`.

### Arguments
- `X::Array{<:Number,2}`: the augmented features, one row per example: the features,
  followed by a 1 for the bias. For `n` examples with `m` features, `X` is `n × (m+1)`.
- `y::Array{<:Number,1}`: the label of each example, `1` or `-1`, in the same order as
  the rows of `X`.
- `α::Float64`: the learning rate, a positive number. Required.
- `β::Float64 = 1.0`: the inverse temperature, a positive number.
- `δ::Float64 = 0.0`: the regularization parameter, at least zero.
- `ϵ::Float64 = 1e-6`: the stopping tolerance on the change in `θ`, a positive number.
- `maxiter::Int64 = 50_000`: the maximum number of gradient descent steps. At least 1.

### Returns
A `NamedTuple` with the fields:
- `θ::Array{Float64,1}`: the estimated parameters, one per column of `X`: the feature
  weights, followed by the bias.
- `iterations::Int64`: the number of steps taken, at most `maxiter`.
- `converged::Bool`: `true` when the last step changed `θ` by at most `ϵ`.
- `losses::Array{Float64,1}`: the regularized loss before the first step and after each
  step, so it has `iterations + 1` entries.

### Errors
Throws an `ArgumentError` if `X` has no rows, a label is not `1` or `-1`, `α`, `β`, or
`ϵ` is not positive, `δ` is negative, or `maxiter < 1`, and a `DimensionMismatch` if `y`
does not have one entry per row of `X`.

### Example
```julia
y = digit_labels(training.labels, 3); # 1 for each image of a 3, -1 for the rest
α = 1/opnorm(training.X)^2; # β = 1, so L = ‖X‖₂²
result = my_logistic_regression(training.X, y, α = α, maxiter = 1000);
result.θ, result.iterations, result.converged
```
"""
function my_logistic_regression(X::Array{<:Number,2}, y::Array{<:Number,1}; α::Float64,
    β::Float64 = 1.0, δ::Float64 = 0.0, ϵ::Float64 = 1e-6,
    maxiter::Int64 = 50_000)::NamedTuple

    # Check the arguments -
    number_of_examples = size(X,1); # the number of training examples, one per row of X
    if number_of_examples < 1
        throw(ArgumentError("X must have at least one row"))
    end
    if length(y) != number_of_examples
        throw(DimensionMismatch("y must have one entry per row of X"))
    end
    if !all((y .== 1) .| (y .== -1)) # .| is "or", applied entry by entry
        throw(ArgumentError("every label in y must be 1 or -1"))
    end
    if !(α > 0 && β > 0 && ϵ > 0)
        throw(ArgumentError("α, β, and ϵ must be positive"))
    end
    if δ < 0
        throw(ArgumentError("δ must be at least zero"))
    end
    if maxiter < 1
        throw(ArgumentError("maxiter must be at least 1"))
    end

    # Initialize -
    θ = zeros(size(X,2)); # the initial guess θ⁽⁰⁾ = 0: the weights, then the bias
    u = 2*β .* y .* (X*θ); # uᵢ = 2β yᵢ x̂ᵢᵀθ, for every example at once
    losses = [loss_from_margins(u, θ, δ)]; # the loss before the first step

    # Main loop: one gradient descent step for each value of k -
    for k ∈ 1:maxiter

        # Step 1: the gradient of the regularized loss at θ, for all examples at once.
        # Row i of X times (1 - σ(uᵢ)) yᵢ, summed over i, is X' * ((1 .- σ.(u)) .* y) -
        ∇J = -2*β .* (X' * ((1 .- σ.(u)) .* y)) .+ δ .* θ;

        # Step 2: the update, θ ← θ - α∇J -
        θ_new = θ .- α .* ∇J;
        u = 2*β .* y .* (X*θ_new); # the new uᵢ, reused by the next gradient
        push!(losses, loss_from_margins(u, θ_new, δ));

        # Step 3: how far did the parameters move? -
        change = norm(θ_new - θ); # the Euclidean norm of the step
        θ = θ_new; # carry the new parameters into the next step

        # The stopping test: the step was at most ϵ, or k has reached maxiter -
        if (change ≤ ϵ || k ≥ maxiter)
            return (θ = θ, iterations = k, converged = (change ≤ ϵ), losses = losses);
        end
    end
end


"""
    σ(z::Real) -> Float64

The logistic function, `σ(z) = 1/(1 + exp(-z))`, from the L9c lecture. It maps any real
number to the interval `(0, 1)`.
"""
σ(z::Real) = 1/(1 + exp(-z))


"""
    loss_from_margins(u, θ, δ) -> Float64

The regularized cross-entropy loss `Σᵢ log(1 + exp(-uᵢ)) + (δ/2)‖θ‖²`, given every
`uᵢ = 2β yᵢ x̂ᵢᵀθ` in the vector `u`.

For a large negative `uᵢ`, `exp(-uᵢ)` is too big for a computer number and becomes
`Inf`. So when `uᵢ < 0`, the term is computed as `-uᵢ + log(1 + exp(uᵢ))`, which has
the same value but only takes `exp` of a negative number.
"""
function loss_from_margins(u, θ, δ)

    # The cross-entropy term of each example -
    terms = zeros(length(u));
    for i ∈ 1:length(u)
        if u[i] < 0
            terms[i] = -u[i] + log1p(exp(u[i])); # the same value as log(1 + exp(-uᵢ)), without overflow
        else
            terms[i] = log1p(exp(-u[i]));        # log1p(z) is log(1 + z), accurate for small z
        end
    end

    # Add the terms, then the regularization term (δ/2)‖θ‖² -
    loss = sum(terms) + (δ/2)*dot(θ, θ);

    return loss;
end
