###########################################################################################################
# This code is computing the koopman operator for the logistic map
# Convergence is exponential in the number of data point, dictionary size grows with number of data points
# Author : Daniel Fassler
###########################################################################################################
include(joinpath(@__DIR__, "..", "koopman_helper_functions.jl"))

# Fix the seed
using Random, CairoMakie
Random.seed!(42)

# Parameters for the stochastic logistic map
p = [2]
m_list = Int.(ceil.(LinRange(2, 30, 18)))
N = 1

scaling(x) = Int(ceil(x^2*log(x)))
n_list = 3*scaling.(m_list)

N_trials = 20
n = 3

# Jacobi parameters
α = 1
β = 0

L2_residuals_unif_leg = zeros(length(n_list), N_trials)
L2_residuals_unif_cheb = zeros(length(n_list), N_trials)
L2_residuals_cheb_leg = zeros(length(n_list), N_trials)
L2_residuals_cheb_cheb = zeros(length(n_list), N_trials)
L2_residuals_unif_mon = zeros(length(n_list), N_trials)
L2_residuals_cheb_mon = zeros(length(n_list), N_trials)

for i = 1:N_trials
    for (j, m) in enumerate(m_list)
        @info "m = $m, trial = $i"

        l = m
        dict_Ψ_Leg = Legendre1(m)
        dict_Φ_Leg = Legendre1(l)
        dict_Ψ_Cheb = Chebyshev1(m)
        dict_Φ_Cheb = Chebyshev1(l)
        dict_Ψ_mon = mon1(m)
        dict_Φ_mon = mon1(l)


        ΨX, ΦY, K_unif_leg = koopman(uniform_sampling_sym, shifted_logistic, Int(n), N, dict_Ψ_Leg, dict_Φ_Leg, p; scale = scaling, verbose = true)
        ΨX, ΦY, K_unif_cheb = koopman(uniform_sampling_sym, shifted_logistic, Int(n), N, dict_Ψ_Cheb, dict_Φ_Cheb, p; scale = scaling)
        ΨX, ΦY, K_cheb_leg = koopman(cheb_sampling_sym, shifted_logistic, Int(n), N, dict_Ψ_Leg, dict_Φ_Leg, p; scale = scaling)
        ΨX, ΦY, K_cheb_cheb = koopman(cheb_sampling_sym, shifted_logistic, Int(n), N, dict_Ψ_Cheb, dict_Φ_Cheb, p; scale = scaling)
        ΨX, ΦY, K_unif_mon = koopman(uniform_sampling_sym, shifted_logistic, Int(n), N, dict_Ψ_mon, dict_Φ_mon, p; scale = scaling)
        ΨX, ΦY, K_cheb_mon = koopman(cheb_sampling_sym, shifted_logistic, Int(n), N, dict_Ψ_mon, dict_Φ_mon, p; scale = scaling)

        L2_residuals_unif_leg[j, i] = L₂_error_Legendre1(shifted_logistic, K_unif_leg, 10*Int(3*m^2*ceil(log(m))), uniform_sampling_sym, p, dict_Ψ_Leg)
        L2_residuals_unif_cheb[j, i] = L₂_error_Chebyshev1(shifted_logistic, K_unif_cheb, 10*Int(3*m^2*ceil(log(m))), uniform_sampling_sym, p, dict_Ψ_Cheb)
        L2_residuals_cheb_leg[j, i] = L₂_error_Legendre1(shifted_logistic, K_cheb_leg, 10*Int(3*m^2*ceil(log(m))), cheb_sampling_sym, p, dict_Ψ_Leg)
        L2_residuals_cheb_cheb[j, i] = L₂_error_Chebyshev1(shifted_logistic, K_cheb_cheb, 10*Int(3*m^2*ceil(log(m))), cheb_sampling_sym, p, dict_Ψ_Cheb)
        L2_residuals_unif_mon[j, i] = L₂_error_Monomial1(shifted_logistic, K_unif_mon, 10*Int(3*m^2*ceil(log(m))), uniform_sampling_sym, p, dict_Ψ_mon)
        L2_residuals_cheb_mon[j, i] = L₂_error_Monomial1(shifted_logistic, K_cheb_mon, 10*Int(3*m^2*ceil(log(m))), cheb_sampling_sym, p, dict_Ψ_mon)
    end
end

res_avg_L2_unif_leg, res_std_L2_unif_leg = log_avg_std(L2_residuals_unif_leg)
res_avg_L2_unif_cheb, res_std_L2_unif_cheb = log_avg_std(L2_residuals_unif_cheb)
res_avg_L2_cheb_leg, res_std_L2_cheb_leg = log_avg_std(L2_residuals_cheb_leg)
res_avg_L2_cheb_cheb, res_std_L2_cheb_cheb = log_avg_std(L2_residuals_cheb_cheb)
res_avg_L2_unif_mon, res_std_L2_unif_mon = log_avg_std(L2_residuals_unif_mon)
res_avg_L2_cheb_mon, res_std_L2_cheb_mon = log_avg_std(L2_residuals_cheb_mon)

# Saving
save(joinpath(@__DIR__, "logistic_exp_data.jld2"), 
    "n_list", n_list, 
    "m_list", m_list,
    "L2_residuals_unif_leg", L2_residuals_unif_leg,
    "L2_residuals_unif_cheb", L2_residuals_unif_cheb,
    "L2_residuals_cheb_leg", L2_residuals_cheb_leg,
    "L2_residuals_cheb_cheb", L2_residuals_cheb_cheb,
    "L2_residuals_unif_mon", L2_residuals_unif_mon,
    "L2_residuals_cheb_mon", L2_residuals_cheb_mon,
    )

# Plotting L2 errors
fig = Figure(size = (1200, 600), backgroundcolor = :white)
a = 0.004
trend_label = "exp(-$(Int(1000*a)) * 10⁻³ * n)"
size_of_labels = 28

# Uniform sampling
ax1 = Axis(fig[1, 1], title = "Uniform Sampling", xlabel = "n", ylabel = "Error", yscale = log10,
    titlesize = 30, xlabelsize = 28, ylabelsize = 28,
    xticklabelsize = size_of_labels, yticklabelsize = size_of_labels)
lines!(ax1, n_list, 10 .^res_avg_L2_unif_leg[:], color = :blue, label = "Legendre", linewidth = 4)
band!(ax1, n_list, 10 .^(res_avg_L2_unif_leg[:] .+ res_std_L2_unif_leg[:]), 10 .^(res_avg_L2_unif_leg[:] .- res_std_L2_unif_leg[:]), color = :blue, alpha = 0.22)
lines!(ax1, n_list, 10 .^res_avg_L2_unif_cheb[:], color = :green, label = "Chebyshev", linewidth = 4)
band!(ax1, n_list, 10 .^(res_avg_L2_unif_cheb[:] .+ res_std_L2_unif_cheb[:]), 10 .^(res_avg_L2_unif_cheb[:] .- res_std_L2_unif_cheb[:]), color = :green, alpha = 0.22)
lines!(ax1, n_list, 10 .^res_avg_L2_unif_mon[:], color = :black, label = "Monomial", linewidth = 4)
band!(ax1, n_list, 10 .^(res_avg_L2_unif_mon[:] .+ res_std_L2_unif_mon[:]), 10 .^(res_avg_L2_unif_mon[:] .- res_std_L2_unif_mon[:]), color = :black, alpha = 0.22)
lines!(ax1, n_list, exp.(-a .* n_list), color = :red, linestyle = :dash, linewidth = 4, label = trend_label)
axislegend(ax1, position = :rt , labelsize = size_of_labels)

# Chebyshev sampling
ax2 = Axis(fig[1, 2], title = "Chebyshev Sampling", xlabel = "n", ylabel = "Error", yscale = log10,
    titlesize = 30, xlabelsize = 28, ylabelsize = 28,
    xticklabelsize = size_of_labels, yticklabelsize = size_of_labels)
lines!(ax2, n_list, 10 .^res_avg_L2_cheb_leg[:], color = :turquoise, label = "Legendre", linewidth = 4)
band!(ax2, n_list, 10 .^(res_avg_L2_cheb_leg[:] .+ res_std_L2_cheb_leg[:]), 10 .^(res_avg_L2_cheb_leg[:] .- res_std_L2_cheb_leg[:]), color = :turquoise, alpha = 0.22)
lines!(ax2, n_list, 10 .^res_avg_L2_cheb_cheb[:], color = :purple, label = "Chebyshev", linewidth = 4)
band!(ax2, n_list, 10 .^(res_avg_L2_cheb_cheb[:] .+ res_std_L2_cheb_cheb[:]), 10 .^(res_avg_L2_cheb_cheb[:] .- res_std_L2_cheb_cheb[:]), color = :purple, alpha = 0.22)
lines!(ax2, n_list, 10 .^res_avg_L2_cheb_mon[:], color = :pink, label = "Monomial", linewidth = 4)
band!(ax2, n_list, 10 .^(res_avg_L2_cheb_mon[:] .+ res_std_L2_cheb_mon[:]), 10 .^(res_avg_L2_cheb_mon[:] .- res_std_L2_cheb_mon[:]), color = :pink, alpha = 0.22)
lines!(ax2, n_list, exp.(-a .* n_list), color = :red, linestyle = :dash, linewidth = 4, label = trend_label)
axislegend(ax2, position = :rt, labelsize = size_of_labels)

save(joinpath(@__DIR__, "logistic_exp.png"), fig)
display(fig)

