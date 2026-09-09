#############################################################################################################################
# This code is computing the koopman operator for the  logistic map
# It showcases the Monte Carlo convergence of the koopman operator for different sampling strategies and basis functions
# Plotting is handled in monte_carlo_plotting.jl
# Author : Daniel Fassler
#############################################################################################################################

# Fix the seed
using Random 
Random.seed!(42)

include(joinpath(@__DIR__, "..", "koopman_helper_functions.jl"))

# Parameters for the  logistic map
p = [2]
n_list = Int.(ceil.(logrange(1e2, 1e5, 20)))
N = 1

m = 5
l = 5
N_trials = 20

dict_Ψ_Leg = Legendre1(m)
dict_Φ_Leg = Legendre1(l)
dict_Ψ_Cheb = Chebyshev1(m)
dict_Φ_Cheb = Chebyshev1(l)
dict_Ψ_mon = mon1(m)
dict_Φ_mon = mon1(l)
dict_Ψ_fou = fourier1(m)
dict_Φ_fou = fourier1(l)




residuals_unif_leg = zeros(length(n_list), N_trials)
residuals_unif_cheb = zeros(length(n_list), N_trials)
residuals_cheb_leg = zeros(length(n_list), N_trials)
residuals_cheb_cheb = zeros(length(n_list), N_trials)
residuals_unif_mon = zeros(length(n_list), N_trials)
residuals_unif_fou = zeros(length(n_list), N_trials)
residuals_cheb_mon = zeros(length(n_list), N_trials)
residuals_cheb_fou = zeros(length(n_list), N_trials)


@info "Computing High fidelity approximations"

ΨX, ΦY, 𝒦_unif_leg_hf = koopman(uniform_sampling_sym, shifted_logistic, Int(1e6), N, dict_Ψ_Leg, dict_Φ_Leg, p ; verbose = true)
ΨX, ΦY, 𝒦_unif_cheb_hf = koopman(uniform_sampling_sym, shifted_logistic, Int(1e6), N, dict_Ψ_Cheb, dict_Φ_Cheb, p ; verbose = true)
ΨX, ΦY, 𝒦_cheb_leg_hf = koopman(cheb_sampling_sym, shifted_logistic, Int(1e6), N, dict_Ψ_Leg, dict_Φ_Leg, p ; verbose = true)
ΨX, ΦY, 𝒦_cheb_cheb_hf = koopman(cheb_sampling_sym, shifted_logistic, Int(1e6), N, dict_Ψ_Cheb, dict_Φ_Cheb, p ; verbose = true)
ΨX, ΦY, 𝒦_unif_mon_hf = koopman(uniform_sampling_sym, shifted_logistic, Int(1e6), N, dict_Ψ_mon, dict_Φ_mon, p ; verbose = true)
ΨX, ΦY, 𝒦_unif_fou_hf = koopman(uniform_sampling_sym, shifted_logistic, Int(1e6), N, dict_Ψ_fou, dict_Φ_fou, p ; verbose = true)
ΨX, ΦY, 𝒦_cheb_mon_hf = koopman(cheb_sampling_sym, shifted_logistic, Int(1e6), N, dict_Ψ_mon, dict_Φ_mon, p ; verbose = true)
ΨX, ΦY, 𝒦_cheb_fou_hf = koopman(cheb_sampling_sym, shifted_logistic, Int(1e6), N, dict_Ψ_fou, dict_Φ_fou, p ; verbose = true)


for i = 1:N_trials
    for (j, m) in enumerate(n_list)
        @info "m = $m, trial = $i"
        ΨX, ΦY, K_unif_leg = koopman(uniform_sampling_sym, shifted_logistic, Int(m), N, dict_Ψ_Leg, dict_Φ_Leg, p)
        ΨX, ΦY, K_unif_cheb = koopman(uniform_sampling_sym, shifted_logistic, Int(m), N, dict_Ψ_Cheb, dict_Φ_Cheb, p)
        ΨX, ΦY, K_cheb_leg = koopman(cheb_sampling_sym, shifted_logistic, Int(m), N, dict_Ψ_Leg, dict_Φ_Leg, p)
        ΨX, ΦY, K_cheb_cheb = koopman(cheb_sampling_sym, shifted_logistic, Int(m), N, dict_Ψ_Cheb, dict_Φ_Cheb, p)
        ΨX, ΦY, K_unif_mon = koopman(uniform_sampling_sym, shifted_logistic, Int(m), N, dict_Ψ_mon, dict_Φ_mon, p)
        ΨX, ΦY, K_unif_fou = koopman(uniform_sampling_sym, shifted_logistic, Int(m), N, dict_Ψ_fou, dict_Φ_fou, p)
        ΨX, ΦY, K_cheb_mon = koopman(cheb_sampling_sym, shifted_logistic, Int(m), N, dict_Ψ_mon, dict_Φ_mon, p)
        ΨX, ΦY, K_cheb_fou = koopman(cheb_sampling_sym, shifted_logistic, Int(m), N, dict_Ψ_fou, dict_Φ_fou, p)

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
save(joinpath(@__DIR__, "logistic_data.jld2"), 
    "n_list", n_list, 
    "residuals_unif_leg", residuals_unif_leg, 
    "residuals_unif_cheb", residuals_unif_cheb, 
    "residuals_cheb_leg", residuals_cheb_leg, 
    "residuals_cheb_cheb", residuals_cheb_cheb,  
    "residuals_unif_mon", residuals_unif_mon, 
    "residuals_unif_fou", residuals_unif_fou, 
    "residuals_cheb_mon", residuals_cheb_mon, 
    "residuals_cheb_fou", residuals_cheb_fou)
