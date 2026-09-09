###########################################################################################################################
# This code is computing the koopman operator for the Thomas model
# It showcases the Monte Carlo convergence of the koopman operator for different sampling strategies and basis functions
# Plotting is handled in monte_carlo_plotting.jl
# Author : Daniel Fassler
###########################################################################################################################
include(joinpath(@__DIR__, "..", "koopman_helper_functions.jl"))

# Fix the seed
using Random 
Random.seed!(42)

# Parameters for the Thomas model
p = [0.208186, 0.01]
n_list = Int.(ceil.(logrange(5 * 1e2, 1e5, 10)))
N = 3
N_trials = 10

# Here m and l are not dictionary size, but rather the degree of the hyperbolic cross considered.
m = 3
l = 3
dict_Ψ_Leg = Legendre3_hyp(m)
dict_Φ_Leg = Legendre3_hyp(l)
dict_Ψ_Cheb = Chebyshev3_hyp(m)
dict_Φ_Cheb = Chebyshev3_hyp(l)
dict_Ψ_mon = mon3_hyp(m)
dict_Φ_mon = mon3_hyp(l)
dict_Ψ_fou = fourier3_hyp(m)
dict_Φ_fou = fourier3_hyp(l)


residuals_unif_leg = zeros(length(n_list), N_trials)
residuals_unif_cheb = zeros(length(n_list), N_trials)
residuals_cheb_leg = zeros(length(n_list), N_trials)
residuals_cheb_cheb = zeros(length(n_list), N_trials)
residuals_unif_mon = zeros(length(n_list), N_trials)
residuals_unif_fou = zeros(length(n_list), N_trials)
residuals_cheb_mon = zeros(length(n_list), N_trials)
residuals_cheb_fou = zeros(length(n_list), N_trials)


@info "Computing High fidelity approximations"

ΨX, ΦY, 𝒦_unif_leg_hf = koopman(uniform_sampling_sym, thomas, Int(1e6), N, dict_Ψ_Leg, dict_Φ_Leg, p ; verbose = true)
ΨX, ΦY, 𝒦_unif_cheb_hf = koopman(uniform_sampling_sym, thomas, Int(1e6), N, dict_Ψ_Cheb, dict_Φ_Cheb, p ; verbose = true)
ΨX, ΦY, 𝒦_cheb_leg_hf = koopman(cheb_sampling_sym, thomas, Int(1e6), N, dict_Ψ_Leg, dict_Φ_Leg, p ;  verbose = true)
ΨX, ΦY, 𝒦_cheb_cheb_hf = koopman(cheb_sampling_sym, thomas, Int(1e6), N, dict_Ψ_Cheb, dict_Φ_Cheb, p ; verbose = true)
ΨX, ΦY, 𝒦_unif_mon_hf = koopman(uniform_sampling_sym, thomas, Int(1e6), N, dict_Ψ_mon, dict_Φ_mon, p ; verbose = true)
ΨX, ΦY, 𝒦_unif_fou_hf = koopman(uniform_sampling_sym, thomas, Int(1e6), N, dict_Ψ_fou, dict_Φ_fou, p ; verbose = true)
ΨX, ΦY, 𝒦_cheb_mon_hf = koopman(cheb_sampling_sym, thomas, Int(1e6), N, dict_Ψ_mon, dict_Φ_mon, p ; verbose = true)
ΨX, ΦY, 𝒦_cheb_fou_hf = koopman(cheb_sampling_sym, thomas, Int(1e6), N, dict_Ψ_fou, dict_Φ_fou, p ; verbose = true)

for i = 1:N_trials
    for (j, n) in enumerate(n_list)
        @info "n = $n, trial = $i"
        ΨX, ΦY, K_unif_leg = koopman(uniform_sampling_sym, thomas, Int(n), N, dict_Ψ_Leg, dict_Φ_Leg, p)
        ΨX, ΦY, K_unif_cheb = koopman(uniform_sampling_sym, thomas, Int(n), N, dict_Ψ_Cheb, dict_Φ_Cheb, p)
        ΨX, ΦY, K_cheb_leg = koopman(cheb_sampling_sym, thomas, Int(n), N, dict_Ψ_Leg, dict_Φ_Leg, p)
        ΨX, ΦY, K_cheb_cheb = koopman(cheb_sampling_sym, thomas, Int(n), N, dict_Ψ_Cheb, dict_Φ_Cheb, p)
        ΨX, ΦY, K_unif_mon = koopman(uniform_sampling_sym, thomas, Int(n), N, dict_Ψ_mon, dict_Φ_mon, p)
        ΨX, ΦY, K_unif_fou = koopman(uniform_sampling_sym, thomas, Int(n), N, dict_Ψ_fou, dict_Φ_fou, p)
        ΨX, ΦY, K_cheb_mon = koopman(cheb_sampling_sym, thomas, Int(n), N, dict_Ψ_mon, dict_Φ_mon, p)
        ΨX, ΦY, K_cheb_fou = koopman(cheb_sampling_sym, thomas, Int(n), N, dict_Ψ_fou, dict_Φ_fou, p)

        residuals_unif_leg[j, i] = norm(𝒦_unif_leg_hf - K_unif_leg)/norm(𝒦_unif_leg_hf)
        residuals_unif_cheb[j, i] = norm(𝒦_unif_cheb_hf - K_unif_cheb)/norm(𝒦_unif_cheb_hf)
        residuals_cheb_leg[j, i] = norm(𝒦_cheb_leg_hf - K_cheb_leg)/norm(𝒦_cheb_leg_hf)
        residuals_cheb_cheb[j, i] = norm(𝒦_cheb_cheb_hf - K_cheb_cheb)/norm(𝒦_cheb_cheb_hf)
        residuals_unif_mon[j, i] = norm(𝒦_unif_mon_hf - K_unif_mon)/norm(𝒦_unif_mon_hf)
        residuals_unif_fou[j, i] = norm(𝒦_unif_fou_hf - K_unif_fou)/norm(𝒦_unif_fou_hf)
        residuals_cheb_mon[j, i] = norm(𝒦_cheb_mon_hf - K_cheb_mon)/norm(𝒦_cheb_mon_hf)
        residuals_cheb_fou[j, i] = norm(𝒦_cheb_fou_hf - K_cheb_fou)/norm(𝒦_cheb_fou_hf)
    end
end

# Saving
save(joinpath(@__DIR__, "thomas_data.jld2"), 
    "n_list", n_list, 
    "residuals_unif_leg", residuals_unif_leg, 
    "residuals_unif_cheb", residuals_unif_cheb, 
    "residuals_cheb_leg", residuals_cheb_leg, 
    "residuals_cheb_cheb", residuals_cheb_cheb, 
    "residuals_unif_mon", residuals_unif_mon, 
    "residuals_unif_fou", residuals_unif_fou, 
    "residuals_cheb_mon", residuals_cheb_mon, 
    "residuals_cheb_fou", residuals_cheb_fou)