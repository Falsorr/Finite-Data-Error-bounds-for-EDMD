################################################################
# Generate and plot the undersampling-rate figures for Lorenz 96.
################################################################

using LinearAlgebra, Random, JLD2, CairoMakie, DynamicalSystems, Convex, COSMO
include("lorenz96_helper_functions.jl")


const F = 8
const d = 10
const basis_Φ = generate_legendre_basis(d)
const basis_Ψ = generate_legendre2_basis(d)
const dictionary_size = length(basis_Φ) * length(basis_Ψ)

function generate_basis_pursuit_data(dt, q, σ, trials, filename; seeded=true, seed=1234)
    if seeded
        Random.seed!(seed)
    end
    @info "Generating basis pursuit data for dt = $dt, q = $q, trials = $trials"
    n_list = q .* round.(Int, LinRange(2, dictionary_size / q - 1, 40))
    operator_list = Array{Matrix{Float64}}(undef, length(n_list), trials)
    validation_residuals = zeros(length(n_list), trials)

    for n_index in eachindex(n_list)
        n = n_list[n_index]
        println("QCBP Trials for n = $n")
        for trial in 1:trials
            X, Y = generate_data(n, q, d, dt)
            Ψ, Φ = generate_observables_matrices(X, Y, basis_Φ, basis_Ψ)
            operator = Variable(length(basis_Φ), length(basis_Ψ))
            problem = minimize(norm(operator, 1), [norm(Φ - operator * Ψ) <= σ])
            solve!(problem, COSMO.Optimizer; silent = true)
            operator_list[n_index, trial] = operator.value
        end
    end

    for n_index in eachindex(n_list)
        n = n_list[n_index]
        for trial in 1:trials
            X, Y = generate_data(n, q, d, dt)
            Ψ, Φ = generate_observables_matrices(X, Y, basis_Φ, basis_Ψ)
            validation_residuals[n_index, trial] = norm(Φ - operator_list[n_index, trial] * Ψ, 2)
        end
    end

    undersampling_rate = n_list ./ dictionary_size
    mean_validation_residuals = vec(mean_row(validation_residuals))
    std_validation_residuals = vec(std_row(validation_residuals))
    save(filename, "K_list", operator_list, "validation_residuals", validation_residuals,
        "undersampling_rate", undersampling_rate,
        "mean_validation_residuals_list", mean_validation_residuals,
        "std_validation_residuals_list", std_validation_residuals)

    return undersampling_rate, mean_validation_residuals
end

function generate_lasso_data(dt, q, λ, trials, filename; seeded=true, seed=1234)
    if seeded
        Random.seed!(seed)
    end
    @info "Generating LASSO data for dt = $dt, q = $q, λ = $λ, trials = $trials"
    n_list = q .* round.(Int, LinRange(2, dictionary_size / q - 1, 40))
    operator_list = Array{Matrix{Float64}}(undef, length(n_list), trials)
    validation_residuals = zeros(length(n_list), trials)

    for n_index in eachindex(n_list)
        n = n_list[n_index]
        println("LASSO Trials for n = $n")
        for trial in 1:trials
            X, Y = generate_data(n, q, d, dt)
            Ψ, Φ = generate_observables_matrices(X, Y, basis_Φ, basis_Ψ)
            operator = Variable(length(basis_Φ), length(basis_Ψ))
            problem = minimize(sumsquares(Φ - operator * Ψ) + λ * norm(operator, 1))
            solve!(problem, COSMO.Optimizer; silent = true)
            operator_list[n_index, trial] = operator.value
        end
    end

    for n_index in eachindex(n_list)
        n = n_list[n_index]
        for trial in 1:trials
            X, Y = generate_data(n, q, d, dt)
            Ψ, Φ = generate_observables_matrices(X, Y, basis_Φ, basis_Ψ)
            validation_residuals[n_index, trial] = norm(Φ - operator_list[n_index, trial] * Ψ, 2)
        end
    end

    undersampling_rate = n_list ./ dictionary_size
    mean_validation_residuals = vec(mean_row(validation_residuals))
    std_validation_residuals = vec(std_row(validation_residuals))
    save(filename, "K_list", operator_list, "validation_residuals", validation_residuals,
        "undersampling_rate", undersampling_rate,
        "mean_validation_residuals_list", mean_validation_residuals,
        "std_validation_residuals_list", std_validation_residuals)

    return undersampling_rate, mean_validation_residuals
end

trials = 10
σ = 0.01
λ = 0.1

function load_or_generate_data(generator, filename)
    if isfile(filename)
        data = load(filename)
        return (data["undersampling_rate"], data["mean_validation_residuals_list"])
    end
    return generator()
end

bp_data = [
    load_or_generate_data("BP_data_run_1.jld2") do
        generate_basis_pursuit_data(0.001, 10, σ, trials, "BP_data_run_1.jld2"; seeded=true, seed=1)
    end,
    load_or_generate_data("BP_data_run_2.jld2") do
        generate_basis_pursuit_data(0.001, 5, σ, trials, "BP_data_run_2.jld2"; seeded=true, seed=2)
    end,
    load_or_generate_data("BP_data_run_3.jld2") do
        generate_basis_pursuit_data(0.01, 10, σ, trials, "BP_data_run_3.jld2"; seeded=true, seed=3)
    end,
    load_or_generate_data("BP_data_run_4.jld2") do
        generate_basis_pursuit_data(0.01, 5, σ, trials, "BP_data_run_4.jld2"; seeded=true, seed=4)
    end,
]

lasso_data = [
    load_or_generate_data("LASSO_data_run_1.jld2") do
        generate_lasso_data(0.001, 10, λ, trials, "LASSO_data_run_1.jld2"; seeded=true, seed=11)
    end,
    load_or_generate_data("LASSO_data_run_2.jld2") do
        generate_lasso_data(0.001, 5, λ, trials, "LASSO_data_run_2.jld2"; seeded=true, seed=12)
    end,
    load_or_generate_data("LASSO_data_run_3.jld2") do
        generate_lasso_data(0.01, 10, λ, trials, "LASSO_data_run_3.jld2"; seeded=true, seed=13)
    end,
    load_or_generate_data("LASSO_data_run_4.jld2") do
        generate_lasso_data(0.01, 5, λ, trials, "LASSO_data_run_4.jld2"; seeded=true, seed=14)
    end,
]

fig1 = Figure(size = (400, 400))
ax1 = Axis(fig1[1, 1]; xlabel = "Undersampling rate", ylabel = "Residuals", title = "Basis Pursuit", yscale = log10)
scatterlines!(ax1, bp_data[1][1], bp_data[1][2], color = :blue, marker = :circle, linestyle = :dot, label = "q = 10, dt = 0.001")
scatterlines!(ax1, bp_data[2][1], bp_data[2][2], color = :red, marker = :circle, linestyle = :solid, label = "q = 5, dt = 0.001")
scatterlines!(ax1, bp_data[3][1], bp_data[3][2], color = :black, marker = :xcross, linestyle = :dot, label = "q = 10, dt = 0.01")
scatterlines!(ax1, bp_data[4][1], bp_data[4][2], color = :green, marker = :xcross, linestyle = :dot, label = "q = 5, dt = 0.01")
axislegend(ax1, position = :rt)

fig5 = Figure(size = (400, 400))
ax5 = Axis(fig5[1, 1]; xlabel = "Undersampling rate", ylabel = "Residuals", title = "LASSO", yscale = log10)
scatterlines!(ax5, lasso_data[1][1], lasso_data[1][2], color = :blue, marker = :circle, linestyle = :dot, label = "q = 10, dt = 0.001")
scatterlines!(ax5, lasso_data[2][1], lasso_data[2][2], color = :red, marker = :circle, linestyle = :solid, label = "q = 5, dt = 0.001")
scatterlines!(ax5, lasso_data[3][1], lasso_data[3][2], color = :black, marker = :xcross, linestyle = :dot, label = "q = 10, dt = 0.01")
scatterlines!(ax5, lasso_data[4][1], lasso_data[4][2], color = :green, marker = :xcross, linestyle = :dot, label = "q = 5, dt = 0.01")
axislegend(ax5, position = :rt)

save("BP_undersampling_rate.png", fig1)
save("LASSO_undersampling_rate.png", fig5)
