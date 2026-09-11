#########################################################################################################################
# This code reads the data in logistic_data.jld2, stochastic_logistic_data.jld2, thomas_data.jld2 and generates Figure 1
# Author : Daniel Fassler
#########################################################################################################################
using JLD2, CairoMakie

include(joinpath(@__DIR__, "..", "koopman_helper_functions.jl"))

# Ensuring data files exist, if not, generate them.
if !isfile(joinpath(@__DIR__, "logistic_data.jld2"))
    println("Generating logistic_data.jld2")
    include(joinpath(@__DIR__, "logistic.jl"))
end

if !isfile(joinpath(@__DIR__, "stochastic_logistic_data.jld2"))
    println("Generating stochastic_logistic_data.jld2")
    include(joinpath(@__DIR__, "stochastic_logistic.jl"))
end

if !isfile(joinpath(@__DIR__, "thomas_data.jld2"))
    println("Generating thomas_data.jld2")
    include(joinpath(@__DIR__, "thomas.jl"))
end

function scaled_trend_line(x_values, y_values; exponent = -1 / 2, lift = 1.5)
    return lift * y_values[1] * (x_values ./ x_values[1]).^exponent
end

function load_mc_dataset(file_path)
    data = load(joinpath(@__DIR__, file_path))
    sample_key = haskey(data, "n_list") ? "n_list" : "m_list"

    n_list = data[sample_key]
    residuals_unif_leg = data["residuals_unif_leg"]
    residuals_unif_cheb = data["residuals_unif_cheb"]
    residuals_cheb_leg = data["residuals_cheb_leg"]
    residuals_cheb_cheb = data["residuals_cheb_cheb"]
    residuals_unif_mon = data["residuals_unif_mon"]
    residuals_unif_fou = data["residuals_unif_fou"]
    residuals_cheb_mon = data["residuals_cheb_mon"]
    residuals_cheb_fou = data["residuals_cheb_fou"]

    avg_unif_leg, std_unif_leg = log_avg_std(residuals_unif_leg)
    avg_unif_cheb, std_unif_cheb = log_avg_std(residuals_unif_cheb)
    avg_cheb_leg, std_cheb_leg = log_avg_std(residuals_cheb_leg)
    avg_cheb_cheb, std_cheb_cheb = log_avg_std(residuals_cheb_cheb)
    avg_unif_mon, std_unif_mon = log_avg_std(residuals_unif_mon)
    avg_unif_fou, std_unif_fou = log_avg_std(residuals_unif_fou)
    avg_cheb_mon, std_cheb_mon = log_avg_std(residuals_cheb_mon)
    avg_cheb_fou, std_cheb_fou = log_avg_std(residuals_cheb_fou)

    return (
        n_list = n_list,
        avg_unif_leg = avg_unif_leg, std_unif_leg = std_unif_leg,
        avg_unif_cheb = avg_unif_cheb, std_unif_cheb = std_unif_cheb,
        avg_cheb_leg = avg_cheb_leg, std_cheb_leg = std_cheb_leg,
        avg_cheb_cheb = avg_cheb_cheb, std_cheb_cheb = std_cheb_cheb,
        avg_unif_mon = avg_unif_mon, std_unif_mon = std_unif_mon,
        avg_unif_fou = avg_unif_fou, std_unif_fou = std_unif_fou,
        avg_cheb_mon = avg_cheb_mon, std_cheb_mon = std_cheb_mon,
        avg_cheb_fou = avg_cheb_fou, std_cheb_fou = std_cheb_fou
    )
end

function plot_mc_row!(fig, row, file_path, model_name)
    data = load_mc_dataset(file_path)

    panels = (
        (1, "Legendre Basis", data.avg_unif_leg, data.std_unif_leg, data.avg_cheb_leg, data.std_cheb_leg),
        (2, "Chebyshev Basis", data.avg_unif_cheb, data.std_unif_cheb, data.avg_cheb_cheb, data.std_cheb_cheb),
        (3, "Monomial Basis", data.avg_unif_mon, data.std_unif_mon, data.avg_cheb_mon, data.std_cheb_mon),
        (4, "Fourier Basis", data.avg_unif_fou, data.std_unif_fou, data.avg_cheb_fou, data.std_cheb_fou)
    )

    for panel in panels
        col, title, avg_unif, std_unif, avg_cheb, std_cheb = panel
        ax = Axis(fig[row, col+1], yscale = log10, xscale = log10,
            titlesize = 30, xlabelsize = 28, ylabelsize = 28,
            xticklabelsize = 30, yticklabelsize = 30)
        lines!(ax, data.n_list, 10 .^avg_unif[:], color = :blue, label = "Uniform Sampling")
        band!(ax, data.n_list, 10 .^(avg_unif[:] .+ std_unif[:]), 10 .^(avg_unif[:] .- std_unif[:]), color = :blue, alpha = 0.3)
        lines!(ax, data.n_list, 10 .^avg_cheb[:], color = :green, label = "Chebyshev Sampling")
        band!(ax, data.n_list, 10 .^(avg_cheb[:] .+ std_cheb[:]), 10 .^(avg_cheb[:] .- std_cheb[:]), color = :green, alpha = 0.3)
        lines!(ax, data.n_list, scaled_trend_line(data.n_list, 10 .^avg_unif), color = :black, label = "Trend Line: n⁻¹/²", linestyle = :dash)
        axislegend(ax, position = :rt, labelsize = 25)
    end

    Label(fig[row, 6], model_name, rotation = -pi/2, fontsize = 34, font = :bold, padding = (0, 0, 0, 0), tellwidth = false, tellheight = false)
    Label(fig[row, 1], "Error", rotation = pi/2, fontsize = 38, font = :bold, padding = (0, 0, 0, 0), tellwidth = false, tellheight = false)
end
    
fig = Figure(size = (2500, 1500))
rowgap!(fig.layout, 15)

plot_mc_row!(fig, 2, "logistic_data.jld2", "Logistic Map")
plot_mc_row!(fig, 3, "stochastic_logistic_data.jld2", "Stochastic Logistic Map")
plot_mc_row!(fig, 4, "thomas_data.jld2", "Thomas Model")

for i in 2:5
    Label(fig[5, i], "Number of samples", fontsize = 38, font = :bold, padding = (0, 0, 0, 0), tellwidth = false, tellheight = false)
end

basis_functions = ["Legendre Basis", "Chebyshev Basis", "Monomial Basis", "Fourier Basis"]
for i in 2:5
    Label(fig[1, i], basis_functions[i-1], fontsize = 38, font = :bold, padding = (0, 0, 0, 0), tellwidth = false, tellheight = false)
end

rowsize!(fig.layout, 1, Fixed(50))
rowsize!(fig.layout, 5, Fixed(50))
colsize!(fig.layout, 1, Fixed(120))
colsize!(fig.layout, 6, Fixed(120))

save(joinpath(@__DIR__, "edmd_summary.png"), fig)
display(fig)