###################################################################
# This file contains the helper functions for the Lorenz 96 model
# Author: Daniel Fassler
###################################################################



using Polynomials, SpecialPolynomials, DynamicalSystems, LinearAlgebra, Random

function lorenz96_rule!(du, u, p, t)
    F = p[1]; N = length(u)
    # 3 edge cases
    du[1] = (u[2] - u[N - 1]) * u[N] - u[1] + F
    du[2] = (u[3] - u[N]) * u[1] - u[2] + F
    du[N] = (u[1] - u[N - 2]) * u[N - 1] - u[N] + F
    # then the general case
    for n in 3:(N - 1)
        du[n] = (u[n + 1] - u[n - 2]) * u[n - 1] - u[n] + F
    end
    return nothing # always `return nothing` for in-place form for DynamicalSystems.jl
end

function generate_burst(dt, m, n)  
    # Generate a trajectory (or a burst as called in the main reference)
    # ϵ is the noise level
    x₀ = 1 .- 2*rand(n)
    p₀ = [F]
    ds = CoupledODEs(lorenz96_rule!, x₀, p₀)
    u, t = trajectory(ds, dt*m, Δt = dt)
    return t, u
end

function generate_data(k, m, n, dt)
    # Generate the data matrices X and Y
    X = zeros(k*m, n)
    Y = zeros(k*m, n)
    # Since the DS is autonomous, we do not need t
    for i in 1:k
        t, u = generate_burst(dt, m, n)
        X[(i-1)*m+1:i*m, :] = Matrix(u[1:end-1])
        Y[(i-1)*m+1:i*m, :] = Matrix(u[2:end])
    end
    return X, Y
end

function generate_monomial_basis(N)
    # Generate the monomial basis
    basis = []
    push!(basis, x -> 1.0)
    for i in 1:N
        push!(basis, x -> x[i])
    end
    return basis
end

function generate_monomial2_basis(N)
    # Generate the monomial basis
    basis = []
    push!(basis, x -> 1.0)
    for i in 1:N
        push!(basis, x -> x[i])
        for j in i:N
            push!(basis, x -> x[i]*x[j])
        end
    end
    return basis
end

function legendre_basis()
    p1 = Legendre([0, 1]) * sqrt(3/2)
    p2 = Legendre([0, 0, 1]) * sqrt(5/2)
    p3 = Legendre([0, 0, 0, 1]) * sqrt(7/2)
    p4 = Legendre([0, 0, 0, 0, 1]) * sqrt(9/2)

    return convert.(Polynomial, [p1, p2, p3, p4])
end

const LEG_BASIS = legendre_basis()

function generate_legendre_basis(N)
    # Generate the Legendre basis
    basis = []
    push!(basis, x -> 1.0)
    for i in 1:N
        push!(basis, x -> LEG_BASIS[1](x[i]))
    end
    return basis
end

function generate_legendre2_basis(N)
    # Generate the Legendre basis
    basis = []
    push!(basis, x -> 1.0)
    for i in 1:N
        push!(basis, x -> LEG_BASIS[1](x[i]))
        push!(basis, x -> LEG_BASIS[2](x[i]))
        for j in i:N
            push!(basis, x -> LEG_BASIS[1](x[i])*LEG_BASIS[1](x[j]))
        end
    end
    return basis
end


function generate_observables_matrices(X, Y, basis_Φ, basis_Ψ; ϵ = 0)
    # Generate the EDMD matrices Ψ and Φ
    # ϵ is the noise level. If ϵ = 0, no noise is added
    l₁ = length(basis_Ψ)
    l₂ = length(basis_Φ)
    d = size(X, 1)
    Ψ = zeros(l₁, d)
    Φ = zeros(l₂, d)
    for i in 1:d
        for j in 1:l₁
            Ψ[j, i] = basis_Ψ[j](X[i, :] .+ ϵ*randn()) 
        end
        for j in 1:l₂
            Φ[j, i] = basis_Φ[j](Y[i, :] .+ ϵ*randn()) 
        end
    end
    return Ψ, Φ
end

function threshold_matrix(matrix, threshold)
    result = copy(matrix)
    for i in 1:size(matrix, 1)
        for j in 1:size(matrix, 2)
            if result[i, j] < threshold
                result[i, j] = 0.0
            end
        end
    end
    return result
end

function mean_row(X)
    # Compute the mean of the rows of X
    return sum(X, dims = 2) / size(X, 2)
end

function std_row(X)
    # Compute the standard deviation of the rows of X
    return sqrt.(sum((X .- mean_row(X)).^2, dims = 2) / size(X, 2))
end