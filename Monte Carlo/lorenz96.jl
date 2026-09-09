###########################################################################################################################
# This code computes the Koopman operator for the Lorenz-96 model.
# It generates Monte Carlo convergence data and a four-panel convergence figure
# in the same file.
###########################################################################################################################

include(joinpath(@__DIR__, "..", "koopman_helper_functions.jl"))

using Random
Random.seed!(1)

function scaled_trend_line(x_values, y_values; exponent = -1 / 2, lift = 1.5)
    return lift * y_values[1] * (x_values ./ x_values[1]).^exponent
end

# Scaled Lorenz-96 dynamics for y_i = C x_i.
function lorenz96_rule!(du, u, p, t)
    F = p[1]
    C = p[3]
    dimension = length(u)
    for i in 1:dimension
        du[i] = ((u[mod1(i + 1, dimension)] - u[mod1(i - 2, dimension)]) *
                 u[mod1(i - 1, dimension)]) / C - u[i] + C * F
    end
    return nothing
end

function lorenz96(x, p; state_scale = 1.0)
    dt = p[2]
    scaled_parameters = [p[1], p[2], state_scale]
    system = CoupledODEs(lorenz96_rule!, state_scale .* x, scaled_parameters)
    trajectory_data, _ = trajectory(system, dt, Δt = dt)
    return Vector(trajectory_data[end]) ./ state_scale
end

function uniform_sampling_lorenz96(m, dimension)
    return [1.8 .* rand(dimension) .- 0.9 for _ in 1:m]
end

function cheb_sampling_lorenz96(m, dimension)
    return [0.9 .* cos.(π .* rand(dimension)) for _ in 1:m]
end

# Generate multi-index sets with product(alpha_i + 1) <= p.
function hyperbolic_cross(dimension, degree)
    indices = Vector{Vector{Int}}()
    for index in Iterators.product((0:degree for _ in 1:dimension)...)
        if prod(value + 1 for value in index) <= degree + 1
            push!(indices, collect(index))
        end
    end
    return indices
end

function lorenz96_dictionary(dimension, degree, family)
    dictionary = Function[]
    for index in hyperbolic_cross(dimension, degree)
        if family == :legendre
            one_dimensional = [begin
                coordinates = zeros(index[i] + 1)
                coordinates[end] = 1
                sqrt((2 * index[i] + 1) / 2) * SP.Legendre(coordinates)
            end for i in 1:dimension]
            push!(dictionary, let one_dimensional = one_dimensional
                x -> prod(one_dimensional[i](x[i]) for i in 1:dimension)
            end)
        elseif family == :chebyshev
            one_dimensional = [begin
                coordinates = zeros(index[i] + 1)
                coordinates[end] = 1
                if index[i] == 0
                    ChebyshevT(coordinates) / sqrt(π)
                else
                    ChebyshevT(coordinates) / sqrt(π / 2)
                end
            end for i in 1:dimension]
            push!(dictionary, let one_dimensional = one_dimensional
                x -> prod(one_dimensional[i](x[i]) for i in 1:dimension)
            end)
        elseif family == :monomial
            one_dimensional = [begin
                coordinates = zeros(index[i] + 1)
                coordinates[end] = 1
                Polynomial(coordinates)
            end for i in 1:dimension]
            push!(dictionary, let one_dimensional = one_dimensional
                x -> prod(one_dimensional[i](x[i]) for i in 1:dimension)
            end)
        elseif family == :fourier
            one_dimensional = [fourier1(degree)[index[i] + 1] for i in 1:dimension]
            push!(dictionary, let one_dimensional = one_dimensional
                x -> prod(one_dimensional[i]([x[i]]) for i in 1:dimension)
            end)
        else
            throw(ArgumentError("Unknown basis family: $family"))
        end
    end
    return dictionary
end

# Model and Monte Carlo parameters.
dimension = 10
forcing = 8.0
dt = 0.001
parameters = [forcing, dt]

n_list = Int.(ceil.(logrange(5 * 1e2, 1e5, 10)))
N_trials = 10
high_fidelity_samples = Int(1e6)

# These are hyperbolic-cross degrees, not dictionary sizes.
degree_Ψ = 1
degree_Φ = 2
dictionaries = Dict(
    family => (lorenz96_dictionary(dimension, degree_Ψ, family),
               lorenz96_dictionary(dimension, degree_Φ, family))
    for family in (:legendre, :chebyshev, :monomial, :fourier)
)

dict_Ψ_Leg, dict_Φ_Leg = dictionaries[:legendre]
dict_Ψ_Cheb, dict_Φ_Cheb = dictionaries[:chebyshev]
dict_Ψ_mon, dict_Φ_mon = dictionaries[:monomial]
dict_Ψ_fou, dict_Φ_fou = dictionaries[:fourier]

residuals_unif_leg = zeros(length(n_list), N_trials)
residuals_unif_cheb = zeros(length(n_list), N_trials)
residuals_cheb_leg = zeros(length(n_list), N_trials)
residuals_cheb_cheb = zeros(length(n_list), N_trials)
residuals_unif_mon = zeros(length(n_list), N_trials)
residuals_unif_fou = zeros(length(n_list), N_trials)
residuals_cheb_mon = zeros(length(n_list), N_trials)
residuals_cheb_fou = zeros(length(n_list), N_trials)

@info "Computing high-fidelity Lorenz-96 approximations"
ΨX, ΦY, K_unif_leg_hf = koopman(uniform_sampling_lorenz96, lorenz96, high_fidelity_samples, dimension, dict_Ψ_Leg, dict_Φ_Leg, parameters; verbose = true)
ΨX, ΦY, K_unif_cheb_hf = koopman(uniform_sampling_lorenz96, lorenz96, high_fidelity_samples, dimension, dict_Ψ_Cheb, dict_Φ_Cheb, parameters; verbose = true)
ΨX, ΦY, K_cheb_leg_hf = koopman(cheb_sampling_lorenz96, lorenz96, high_fidelity_samples, dimension, dict_Ψ_Leg, dict_Φ_Leg, parameters; verbose = true)
ΨX, ΦY, K_cheb_cheb_hf = koopman(cheb_sampling_lorenz96, lorenz96, high_fidelity_samples, dimension, dict_Ψ_Cheb, dict_Φ_Cheb, parameters; verbose = true)
ΨX, ΦY, K_unif_mon_hf = koopman(uniform_sampling_lorenz96, lorenz96, high_fidelity_samples, dimension, dict_Ψ_mon, dict_Φ_mon, parameters; verbose = true)
ΨX, ΦY, K_unif_fou_hf = koopman(uniform_sampling_lorenz96, lorenz96, high_fidelity_samples, dimension, dict_Ψ_fou, dict_Φ_fou, parameters; verbose = true)
ΨX, ΦY, K_cheb_mon_hf = koopman(cheb_sampling_lorenz96, lorenz96, high_fidelity_samples, dimension, dict_Ψ_mon, dict_Φ_mon, parameters; verbose = true)
ΨX, ΦY, K_cheb_fou_hf = koopman(cheb_sampling_lorenz96, lorenz96, high_fidelity_samples, dimension, dict_Ψ_fou, dict_Φ_fou, parameters; verbose = true)

for trial in 1:N_trials
    for (sample_index, sample_count) in enumerate(n_list)
        @info "n = $sample_count, trial = $trial"
        ΨX, ΦY, K_unif_leg = koopman(uniform_sampling_lorenz96, lorenz96, sample_count, dimension, dict_Ψ_Leg, dict_Φ_Leg, parameters)
        ΨX, ΦY, K_unif_cheb = koopman(uniform_sampling_lorenz96, lorenz96, sample_count, dimension, dict_Ψ_Cheb, dict_Φ_Cheb, parameters)
        ΨX, ΦY, K_cheb_leg = koopman(cheb_sampling_lorenz96, lorenz96, sample_count, dimension, dict_Ψ_Leg, dict_Φ_Leg, parameters)
        ΨX, ΦY, K_cheb_cheb = koopman(cheb_sampling_lorenz96, lorenz96, sample_count, dimension, dict_Ψ_Cheb, dict_Φ_Cheb, parameters)
        ΨX, ΦY, K_unif_mon = koopman(uniform_sampling_lorenz96, lorenz96, sample_count, dimension, dict_Ψ_mon, dict_Φ_mon, parameters)
        ΨX, ΦY, K_unif_fou = koopman(uniform_sampling_lorenz96, lorenz96, sample_count, dimension, dict_Ψ_fou, dict_Φ_fou, parameters)
        ΨX, ΦY, K_cheb_mon = koopman(cheb_sampling_lorenz96, lorenz96, sample_count, dimension, dict_Ψ_mon, dict_Φ_mon, parameters)
        ΨX, ΦY, K_cheb_fou = koopman(cheb_sampling_lorenz96, lorenz96, sample_count, dimension, dict_Ψ_fou, dict_Φ_fou, parameters)

        residuals_unif_leg[sample_index, trial] = norm(K_unif_leg_hf - K_unif_leg) / norm(K_unif_leg_hf)
        residuals_unif_cheb[sample_index, trial] = norm(K_unif_cheb_hf - K_unif_cheb) / norm(K_unif_cheb_hf)
        residuals_cheb_leg[sample_index, trial] = norm(K_cheb_leg_hf - K_cheb_leg) / norm(K_cheb_leg_hf)
        residuals_cheb_cheb[sample_index, trial] = norm(K_cheb_cheb_hf - K_cheb_cheb) / norm(K_cheb_cheb_hf)
        residuals_unif_mon[sample_index, trial] = norm(K_unif_mon_hf - K_unif_mon) / norm(K_unif_mon_hf)
        residuals_unif_fou[sample_index, trial] = norm(K_unif_fou_hf - K_unif_fou) / norm(K_unif_fou_hf)
        residuals_cheb_mon[sample_index, trial] = norm(K_cheb_mon_hf - K_cheb_mon) / norm(K_cheb_mon_hf)
        residuals_cheb_fou[sample_index, trial] = norm(K_cheb_fou_hf - K_cheb_fou) / norm(K_cheb_fou_hf)
    end
end

output_directory = @__DIR__
save(joinpath(output_directory, "lorenz96_data.jld2"),
    "n_list", n_list,
    "residuals_unif_leg", residuals_unif_leg,
    "residuals_unif_cheb", residuals_unif_cheb,
    "residuals_cheb_leg", residuals_cheb_leg,
    "residuals_cheb_cheb", residuals_cheb_cheb,
    "residuals_unif_mon", residuals_unif_mon,
    "residuals_unif_fou", residuals_unif_fou,
    "residuals_cheb_mon", residuals_cheb_mon,
    "residuals_cheb_fou", residuals_cheb_fou)

# Render one convergence row, matching the four panels used by monte_carlo_plotting.jl.
plot_data = (
    ("Legendre Basis", residuals_unif_leg, residuals_cheb_leg),
    ("Chebyshev Basis", residuals_unif_cheb, residuals_cheb_cheb),
    ("Monomial Basis", residuals_unif_mon, residuals_cheb_mon),
    ("Fourier Basis", residuals_unif_fou, residuals_cheb_fou)
)

figure = Figure(size = (2500, 650))
for (column, (title, uniform_residuals, chebyshev_residuals)) in enumerate(plot_data)
    uniform_average, uniform_std = log_avg_std(uniform_residuals)
    chebyshev_average, chebyshev_std = log_avg_std(chebyshev_residuals)
    axis = Axis(figure[1, column], yscale = log10, xscale = log10,
        title = title, titlesize = 30, xlabel = "Number of samples", ylabel = "Error",
        xlabelsize = 24, ylabelsize = 24, xticklabelsize = 20, yticklabelsize = 20)
    lines!(axis, n_list, 10 .^ uniform_average[:], color = :blue, label = "Uniform Sampling")
    band!(axis, n_list, 10 .^ (uniform_average[:] .+ uniform_std[:]),
        10 .^ (uniform_average[:] .- uniform_std[:]), color = :blue, alpha = 0.3)
    lines!(axis, n_list, 10 .^ chebyshev_average[:], color = :green, label = "Chebyshev Sampling")
    band!(axis, n_list, 10 .^ (chebyshev_average[:] .+ chebyshev_std[:]),
        10 .^ (chebyshev_average[:] .- chebyshev_std[:]), color = :green, alpha = 0.3)
    lines!(axis, n_list, scaled_trend_line(n_list, 10 .^ uniform_average),
        color = :black, linestyle = :dash, label = "Trend Line: n⁻¹/²")
    axislegend(axis, position = :rt, labelsize = 18)
end

Label(figure[0, :], "Lorenz-96 Model (d = 10, F = 8)", fontsize = 36, font = :bold)
save(joinpath(output_directory, "lorenz96_convergence.png"), figure)
display(figure)
