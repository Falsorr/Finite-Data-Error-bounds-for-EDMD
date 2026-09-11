###############################################################################
# This file containes all the helper functions that are used in the EDMD files.
# Author : Daniel Fassler
###############################################################################
using LinearAlgebra, Random, Polynomials, SpecialPolynomials, SparseArrays, JLD2, DynamicalSystems

const SP = SpecialPolynomials


########################## BASIC FUNCTIONS ###########################
    # Sourced from https://discourse.julialang.org/t/create-d-dimensional-grid-with-n-points-in-each-direction-where-d-can-be-higher-than-3/28197/3 by Tamas_Papp and benzwick
    # Generate a d-dimensional grid with n points in each direction
    creategrid(d, n, bounds) = vec(collect(Iterators.product((bounds[1]:(bounds[2] - bounds[1])/max(1, (n-1)):bounds[2] for _ in 1:d)...)))

    function hyperbolic_cross3(n; a = (1,1,1))
        # Generates the 3d hyperbolic cross index set of maximum vertex  \in N
        tensor_grid = vec(collect(Iterators.product(0:n, 0:n, 0:n)))
        hyperbolic_cross = []
        for i in tensor_grid
            if ((i[1]+1)^a[1] * (i[2]+1)^a[2] * (i[3]+1)^a[3]) <= (n+1)
                push!(hyperbolic_cross, i)
            end
        end
        return hyperbolic_cross
    end

    function sample_average(X)
        # X is a matrix where each row correspond to a set of experiment
        # We remove any outliers with Inf values
        avg = zeros(length(X[:,1]), 1)
        N = length(X[1,:])
        for j = 1:length(X[:,1])
            count = 0
            for i = 1:N
                if !isnan(X[j,i])
                    count += 1
                    avg[j] += X[j,i]
                end
            end
            avg[j] /= count
        end
        return avg
    end

    function sample_std_dev(X)
        N = length(X[1,:])
        σ = zeros(length(X[:,1]), 1)
        μ = sample_average(X)
        for j = 1:length(X[:,1])
            count = 0
            for i = 1:N
                if !isnan(X[j,i])
                    count += 1
                    σ[j] += (X[j, i] - μ[j])^2
                end
            end
            σ[j] *= 1/(count-1)
        end
        return sqrt.(σ)
    end

    function log_avg_std(X)
        # computes the average and std in log scale for plotting purposes
        X = log10.(X)
        avg = sample_average(X)
        σ = sample_std_dev(X)
        return avg, σ
    end

    function pad(x, n)
        # pad vector x with zeros to length n
        if length(x) <= n
            return [x; zeros(n - length(x))]
        else
            return x[1:n]
        end
    end

    function MonToLeg(a)
        # Convert monomial coefficients to Legendre coefficients
        l = length(a)
        p = Polynomial(a)
        p_leg = coeffs(convert(Legendre, p))
        p_leg = pad(p_leg, l)

        # Normalize by multiplying the coefficients by sqrt(2/(2i+1))
        # (Equivalent to divide the basis function by sqrt(2/(2i+1)))
        for i = 0:l-1
            p_leg[i+1] = p_leg[i+1] * sqrt(2/(2i+1))
        end

        return p_leg
    end

    function MonToCheb(a)
        # Convert monomial coefficients to ChebyshevT coefficients
        l = length(a)
        p = Polynomial(a)
        p_cheb = coeffs(convert(ChebyshevT, p))
        p_cheb = pad(p_cheb, l)

        # Normalize by multiplying the coefficients by sqrt(π/2) for i > 0 and sqrt(π) for i = 0
        p_cheb[1] = p_cheb[1] * sqrt(π)
        for i = 1:l-1
            p_cheb[i+1] = p_cheb[i+1] * sqrt(π/2)
        end

        return p_cheb
    end
########################## KOOPMAN OPERATOR ##########################
    function koopman(samp, ds, n, N, dict_Ψ, dict_Φ, p; weight = x -> 1, scale = x -> 1, verbose = false)
        # This function generates the data for the Koopman approximation setup
        # samp : sampling strategy
        # ds : dynamical system (In case of continuous time, this should be something like forward euler)
        # n : number of data points
        # N : dimension of the dynamical system
        # dict_Ψ : dictionary of the observable functions
        # dict_Φ : dictionary of the observable functions
        # p : parameters of the dynamical system
        # scale : scaling function
        # morph, morph_inv : For some cases, the domain of the dynamical system is not always the same as the domain of the observable functions,
        # morph and morph_inv are used to morph the domain of the dynamical system to the domain of the observable functions
        m = length(dict_Ψ)
        l = length(dict_Φ)
        n_scaled = scale(m)*n

        if verbose == true
            @info "Summary of simulation"
            println("\tNumber of observable functions (Ψ): \t", m)
            println("\tNumber of observable functions (Φ): \t", l)
            println("\tNumber of data points: \t\t", n_scaled)
            println("\tDimensions of the dynamical system: \t", N)
        end

        # Generate data
        if verbose == true
            @info "Generating data"
        end
        X = samp(ceil(n_scaled), N)

        Y = Array[]
        for i = 1:ceil(n_scaled)
            # Making sure the input is a vector
            X_tuple = X[i]
            X_array = [i for i in X_tuple]
            push!(Y, ds(X_array, p))
        end


        if verbose == true
            @info "Computing Koopman operator"
        end
        # Build observables matrices
        ΨX = zeros(m, n_scaled)
        ΦY = zeros(l, n_scaled)


        for i = 1:n_scaled
            # Making sure the input is a vector
            X_tuple = X[i]
            X_array = [i for i in X_tuple]

            Y_tuple = Y[i]
            Y_array = [i for i in Y_tuple]
            for j = 1:m
                ΨX[j, i] = dict_Ψ[j](X_array)
            end
            for j = 1:l
                ΦY[j, i] =dict_Φ[j](Y_tuple)
            end
        end

        # Construct the weight matrix for weighted least squares
        W = spzeros(n_scaled, n_scaled)
        for i = 1:n_scaled
            W[i, i] = sqrt(weight(X[i]))
        end
        if verbose == true
            @info "Max weight : $(maximum(diag(W)))"
        end

        # Compute Koopman operator
        𝒦 = ΦY*W*pinv(ΨX*W)
        if verbose == true
            @info "Residual norm: $(norm(ΦY - 𝒦*ΨX))"
        end
        return ΨX, ΦY, 𝒦
    end
########################## SAMPLING FUNCTIONS ########################
    function uniform_sampling_sym(m, N)
        # This function generates m N-dimensional uniform samples in the interval [-1, 1]
        samples = []
        for i = 1:m
            sample = 2*rand(N) .- 1
            push!(samples, sample)
        end
        return samples
    end

    function cheb_sampling_sym(m, N)
        # This function generates m N-dimensional Chebyshev samples in the interval [-1, 1]
        samples = []
        for i = 1:m
            sample = cos.(π*rand(N)) 
            push!(samples, sample)
        end
        return samples
    end

########################## DYNAMICAL SYSTEMS #########################
    function shifted_logistic(x, p)
        # This function computes the shifted logistic map
        λ = p[1]
        return [λ*x[1]^2 - 1]
    end

    function shifted_stochastic_logistic(x, p)
        # This function computes the shifted stochastic logistic map
        return [p[1]*rand()*x[1]^2 - 1]
    end


    function thomas_rule!(du, u, p, t)
        b = p[1]
        du[1] = 0.2*(sin(5*u[2]) .- b*5*u[1])
        du[2] = 0.2*(sin(5*u[3]) .- b*5*u[2])
        du[3] = 0.2*(sin(5*u[1]) .- b*5*u[3])
        return nothing
    end

    function thomas(x, p)
        # This function computes the Thomas cyclically symmetric attractor
        dt = p[2]
        diffeq = (reltol = 1e-7, abstol = 1e-7)
        ds = CoupledODEs(thomas_rule!, x, p)
        u, t = trajectory(ds, dt, Δt = dt)
        return Vector(u[end])
    end

########################## BASIS FUNCTIONS GENERATORS ################
    function Legendre1(p)
        # Generates the first p Legendre polyonmials in the interval [-1, 1] (normalized)
        b = []
        for i = 0:p
            coordinates = zeros(i+1)
            coordinates[i+1] = 1
            pol = sqrt((2i+1) / 2) * SP.Legendre(coordinates)
            push!(b, x -> pol(x[1]))
        end

        return b
    end

    function Chebyshev1(p)
        # Generates the first p Chebyshev polyonmials in the interval [-1, 1] (normalized)
        basis = []
        for i = 0:p
            coordinates = zeros(i+1)
            coordinates[i+1] = 1
            if i == 0
                pol = ChebyshevT(coordinates) / sqrt(π)
            else
                pol = ChebyshevT(coordinates) / sqrt(π/2)
            end
            push!(basis, x -> pol(x[1]))
        end

        return basis
    end

    function mon1(p)
        # Generates the first p+1 monomials in the interval [-1, 1]
        basis = []
        for i = 0:p
            coordinates = zeros(i+1)
            coordinates[i+1] = 1
            pol = Polynomial(coordinates)
            push!(basis, x -> pol(x[1]))
        end
        return basis
    end

    function fourier1(p)
    # Generates the first 2p+1 Fourier basis function sin(aπx), cos(aπx) for a = 1, 2, ..., p
        basis = []
        push!(basis, x -> 1)
        for i = 1:p
            push!(basis, x -> sin(π*i*x[1]))
            push!(basis, x -> cos(π*i*x[1]))
        end
        return basis
    end

    function Legendre3_hyp(p)
    # Generates the multivariate Legendre polynomials in the hyperbolic cross Lambda_p
        basis = []
        for i in hyperbolic_cross3(p)
            coordinates1 = zeros(i[1]+1)
            coordinates1[i[1]+1] = 1
            coordinates2 = zeros(i[2]+1)
            coordinates2[i[2]+1] = 1
            coordinates3 = zeros(i[3]+1)
            coordinates3[i[3]+1] = 1
            pol1 = sqrt((2i[1]+1) / 2) * SP.Legendre(coordinates1)
            pol2 = sqrt((2i[2]+1) / 2) * SP.Legendre(coordinates2)
            pol3 = sqrt((2i[3]+1) / 2) * SP.Legendre(coordinates3)
            push!(basis, x -> pol1(x[1]) * pol2(x[2]) * pol3(x[3]))
        end
        return basis
    end

    function Chebyshev3_hyp(p)
    # Generates the multivariate Chebyshev polynomials in the hyperbolic cross Lambda_p
        basis = []
        for i in hyperbolic_cross3(p)
            coordinates1 = zeros(i[1]+1)
            coordinates1[i[1]+1] = 1
            coordinates2 = zeros(i[2]+1)
            coordinates2[i[2]+1] = 1
            coordinates3 = zeros(i[3]+1)
            coordinates3[i[3]+1] = 1
            if i[1] == 0
                pol1 = ChebyshevT(coordinates1) / sqrt(π)
            else
                pol1 = ChebyshevT(coordinates1) / sqrt(π/2)
            end
            if i[2] == 0
                pol2 = ChebyshevT(coordinates2) / sqrt(π)
            else
                pol2 = ChebyshevT(coordinates2) / sqrt(π/2)
            end
            if i[3] == 0
                pol3 = ChebyshevT(coordinates3) / sqrt(π)
            else
                pol3 = ChebyshevT(coordinates3) / sqrt(π/2)
            end
            push!(basis, x -> pol1(x[1]) * pol2(x[2]) * pol3(x[3]))
        end
        return basis
    end

    function mon3_hyp(p)
    # Generates the multivariate monomials in the hyperbolic cross Lambda_p
        basis = []
        for i in hyperbolic_cross3(p)
            coordinates1 = zeros(i[1]+1)
            coordinates1[i[1]+1] = 1
            coordinates2 = zeros(i[2]+1)
            coordinates2[i[2]+1] = 1
            coordinates3 = zeros(i[3]+1)
            coordinates3[i[3]+1] = 1
            pol1 = Polynomial(coordinates1)
            pol2 = Polynomial(coordinates2)
            pol3 = Polynomial(coordinates3)
            push!(basis, x -> pol1(x[1]) * pol2(x[2]) * pol3(x[3]))
        end
        return basis
    end

    function fourier3_hyp(p)
    # Generates the multivariate Fourier basis function sin(a\pi x), cos(a\pi x) for a = 1, 2, ..., p in the hyperbolic cross Lambda_(2p+1)
        basis = []
        list = fourier1(p)
        n = length(list) - 1
        for i in hyperbolic_cross3(n)
            push!(basis, x -> list[i[1]+1](x[1]) * list[i[2]+1](x[2]) * list[i[3]+1](x[3]))
        end
        return basis
    end


########################## ERROR COMPUTATION #########################

    function L₂_error_Legendre1(ds, A, m, sampling, p, dict_ψ)
        # Compute the L₂ error of the approximation A[ϕ] of 𝒦[ϕ], for ϕ an observable closed under the action of the Koopman operator.
        function 𝒦ϕ(x) 
            val = 0
            for i = 1:20
                val += exp(ds(x, p)[1])
            end
            return val / 20
        end
        l,n = size(A)
        c = EXP_LEG[1:l]'
        # Getting the coefficients of the Legendre Polynomial approximation of 𝒦f
        α = c*A
        function 𝒦ϕhat(x)
            sum = 0
            for i = 1:l
                sum += α[i]*dict_ψ[i](x)
            end
            return sum
        end
        y = sampling(m, 1)
        sum1 = 0
        sum2 = 0
        for j = 1:m
            sum1 += 0.5*(𝒦ϕ(y[j]) - 𝒦ϕhat(y[j]))^2
            sum2 += 𝒦ϕ(y[j])^2
        end
        error = sqrt(sum1/sum2)
        return error
    end

    function L₂_error_Chebyshev1(ds, A, m, sampling, p, dict_ψ)
        # Compute the L₂ error of the approximation A[ϕ] of 𝒦[ϕ], for ϕ an observable closed under the action of the Koopman operator.
        function 𝒦ϕ(x)
            val = 0
            for i = 1:20
                val += exp(ds(x, p)[1])
            end
            return val / 20
        end
        l, n = size(A)
        c = EXP_CHEB[1:l]'
        # Getting the coefficients of the Chebyshev Polynomial approximation of 𝒦f
        α = c * A

        function 𝒦ϕhat(x)
            sum = 0
            for i = 1:l
                sum += α[i] * dict_ψ[i](x)
            end
            return sum
        end
        y = sampling(m, 1)
        sum1 = 0
        sum2 = 0
        for j = 1:m
            sum1 += 1/(π*sqrt(1 - y[j][1])) * (𝒦ϕ(y[j]) - 𝒦ϕhat(y[j]))^2
            sum2 += 1/(π*sqrt(1 - y[j][1])) * 𝒦ϕ(y[j])^2
        end
        error = sqrt(sum1 / sum2)
        return error
    end

    function L₂_error_Monomial1(ds, A, m, sampling, p, dict_ψ)
        # Compute the L₂ error of the approximation A[ϕ] of 𝒦[ϕ], for ϕ an observable closed under the action of the Koopman operator.
        function 𝒦ϕ(x)
            val = 0
            for i = 1:20
                val += exp(ds(x, p)[1])
            end
            return val / 20
        end
        l, n = size(A)
        c = EXP_MON[1:l]'
        # Getting the coefficients of the Monomial approximation of 𝒦f
        α = c * A

        function 𝒦ϕhat(x)
            sum = 0
            for i = 1:l
                sum += α[i] * dict_ψ[i](x)
            end
            return sum
        end
        y = sampling(m, 1)
        sum1 = 0
        sum2 = 0
        for j = 1:m
            sum1 += (𝒦ϕ(y[j]) - 𝒦ϕhat(y[j]))^2
            sum2 += 𝒦ϕ(y[j])^2
        end
        error = sqrt(sum1 / sum2)
        return error
    end

########################## CONSTANTS #################################
    const EXP_MON_NON_ADJ = load("EXP_MON_NON_ADJ.jld2")["EXP_MON_NON_ADJ"]
    const EXP_MON = EXP_MON_NON_ADJ'
    const EXP_LEG = MonToLeg(EXP_MON_NON_ADJ)'
    const EXP_CHEB = MonToCheb(EXP_MON_NON_ADJ)'
    const ONE_D_GRID_SYM = creategrid(1, 10000, (-1, 1))
    const TWO_D_GRID_SYM = creategrid(2, 10000, (-1, 1))
    const THREE_D_GRID_SYM = creategrid(3, 100, (-1, 1))