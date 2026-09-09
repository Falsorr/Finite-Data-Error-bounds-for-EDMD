#######################################################################
# This code is computing the koopman operator for the stochastic logistic map
# It showcases the Monte Carlo convergence of the koopman operator for different sampling strategies and basis functions
# Plotting is handled in monte_carlo_plotting.jl
# Author : Daniel Fassler
#######################################################################
include(joinpath(@__DIR__, "..", "koopman_helper_functions.jl"))

# Fix the seed
using Random 
Random.seed!(42)

# Parameters for the stochastic logistic map
p = [2]
m_list = Int.(ceil.(logrange(1e2, 1e5, 20)))
N = 1

# Jacobi parameters
α = 1
β = 0

n = 5
l = 5
dict_Ψ_Leg = Legendre1(n)
dict_Φ_Leg = Legendre1(l)
dict_Ψ_Cheb = Chebyshev1(n)
dict_Φ_Cheb = Chebyshev1(l)
dict_Ψ_mon = mon1(n)
dict_Φ_mon = mon1(l)
dict_Ψ_fou = fourier1(n)
dict_Φ_fou = fourier1(l)


N_trials = 20
N_trials_hf = 10



residuals_unif_leg = zeros(length(m_list), N_trials)
residuals_unif_cheb = zeros(length(m_list), N_trials)
residuals_cheb_leg = zeros(length(m_list), N_trials)
residuals_cheb_cheb = zeros(length(m_list), N_trials)
residuals_unif_mon = zeros(length(m_list), N_trials)
residuals_unif_fou = zeros(length(m_list), N_trials)
residuals_cheb_mon = zeros(length(m_list), N_trials)
residuals_cheb_fou = zeros(length(m_list), N_trials)

𝒦_unif_leg_hf_list = Array{Matrix}(undef, N_trials_hf)
𝒦_unif_cheb_hf_list = Array{Matrix}(undef, N_trials_hf)
𝒦_cheb_leg_hf_list = Array{Matrix}(undef, N_trials_hf)
𝒦_cheb_cheb_hf_list = Array{Matrix}(undef, N_trials_hf)
𝒦_unif_mon_hf_list = Array{Matrix}(undef, N_trials_hf)
𝒦_unif_fou_hf_list = Array{Matrix}(undef, N_trials_hf)
𝒦_cheb_mon_hf_list = Array{Matrix}(undef, N_trials_hf)
𝒦_cheb_fou_hf_list = Array{Matrix}(undef, N_trials_hf)


@info "Computing High fidelity approximations"
for i = 1:N_trials_hf
    ΨX, ΦY, 𝒦_unif_leg_hf_list[i] = koopman(uniform_sampling_sym, shifted_stochastic_logistic, Int(1e6), N, dict_Ψ_Leg, dict_Φ_Leg, p ; verbose = true)
    ΨX, ΦY, 𝒦_unif_cheb_hf_list[i] = koopman(uniform_sampling_sym, shifted_stochastic_logistic, Int(1e6), N, dict_Ψ_Cheb, dict_Φ_Cheb, p ; verbose = true)
    ΨX, ΦY, 𝒦_cheb_leg_hf_list[i] = koopman(cheb_sampling_sym, shifted_stochastic_logistic, Int(1e6), N, dict_Ψ_Leg, dict_Φ_Leg, p ;  verbose = true)
    ΨX, ΦY, 𝒦_cheb_cheb_hf_list[i] = koopman(cheb_sampling_sym, shifted_stochastic_logistic, Int(1e6), N, dict_Ψ_Cheb, dict_Φ_Cheb, p ; verbose = true)
    ΨX, ΦY, 𝒦_unif_mon_hf_list[i] = koopman(uniform_sampling_sym, shifted_stochastic_logistic, Int(1e6), N, dict_Ψ_mon, dict_Φ_mon, p ; verbose = true)
    ΨX, ΦY, 𝒦_unif_fou_hf_list[i] = koopman(uniform_sampling_sym, shifted_stochastic_logistic, Int(1e6), N, dict_Ψ_fou, dict_Φ_fou, p ; verbose = true)
    ΨX, ΦY, 𝒦_cheb_mon_hf_list[i] = koopman(cheb_sampling_sym, shifted_stochastic_logistic, Int(1e6), N, dict_Ψ_mon, dict_Φ_mon, p ; verbose = true)
    ΨX, ΦY, 𝒦_cheb_fou_hf_list[i] = koopman(cheb_sampling_sym, shifted_stochastic_logistic, Int(1e6), N, dict_Ψ_fou, dict_Φ_fou, p ; verbose = true)
end

𝒦_unif_leg_hf = sum(𝒦_unif_leg_hf_list)/N_trials_hf
𝒦_unif_cheb_hf = sum(𝒦_unif_cheb_hf_list)/N_trials_hf
𝒦_cheb_leg_hf = sum(𝒦_cheb_leg_hf_list)/N_trials_hf
𝒦_cheb_cheb_hf = sum(𝒦_cheb_cheb_hf_list)/N_trials_hf
𝒦_unif_mon_hf = sum(𝒦_unif_mon_hf_list)/N_trials_hf
𝒦_unif_fou_hf = sum(𝒦_unif_fou_hf_list)/N_trials_hf
𝒦_cheb_mon_hf = sum(𝒦_cheb_mon_hf_list)/N_trials_hf
𝒦_cheb_fou_hf = sum(𝒦_cheb_fou_hf_list)/N_trials_hf

for i = 1:N_trials
    for (j, m) in enumerate(m_list)
        @info "m = $m, trial = $i"
        ΨX, ΦY, K_unif_leg = koopman(uniform_sampling_sym, shifted_stochastic_logistic, Int(m), N, dict_Ψ_Leg, dict_Φ_Leg, p)
        ΨX, ΦY, K_unif_cheb = koopman(uniform_sampling_sym, shifted_stochastic_logistic, Int(m), N, dict_Ψ_Cheb, dict_Φ_Cheb, p)
        ΨX, ΦY, K_cheb_leg = koopman(cheb_sampling_sym, shifted_stochastic_logistic, Int(m), N, dict_Ψ_Leg, dict_Φ_Leg, p)
        ΨX, ΦY, K_cheb_cheb = koopman(cheb_sampling_sym, shifted_stochastic_logistic, Int(m), N, dict_Ψ_Cheb, dict_Φ_Cheb, p)
        ΨX, ΦY, K_unif_mon = koopman(uniform_sampling_sym, shifted_stochastic_logistic, Int(m), N, dict_Ψ_mon, dict_Φ_mon, p)
        ΨX, ΦY, K_unif_fou = koopman(uniform_sampling_sym, shifted_stochastic_logistic, Int(m), N, dict_Ψ_fou, dict_Φ_fou, p)
        ΨX, ΦY, K_cheb_mon = koopman(cheb_sampling_sym, shifted_stochastic_logistic, Int(m), N, dict_Ψ_mon, dict_Φ_mon, p)
        ΨX, ΦY, K_cheb_fou = koopman(cheb_sampling_sym, shifted_stochastic_logistic, Int(m), N, dict_Ψ_fou, dict_Φ_fou, p)

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
save(joinpath(@__DIR__, "stochastic_logistic_data.jld2"), 
    "m_list", m_list, 
    "residuals_unif_leg", residuals_unif_leg, 
    "residuals_unif_cheb", residuals_unif_cheb, 
    "residuals_cheb_leg", residuals_cheb_leg, 
    "residuals_cheb_cheb", residuals_cheb_cheb, 
    "residuals_unif_mon", residuals_unif_mon,
    "residuals_unif_fou", residuals_unif_fou,
    "residuals_cheb_mon", residuals_cheb_mon,
    "residuals_cheb_fou", residuals_cheb_fou
    )