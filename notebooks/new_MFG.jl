using ProgressBars;
using ProgressMeter
using LinearAlgebra, Plots, FFTW;
using LoggingExtras, ProgressLogging, Logging, TerminalLoggers
using Infiltrator
import LinearSolve as LS
using SparseArrays
using QuadGK;
using SpecialFunctions: gamma, loggamma, zeta

# Functions

function central_diff_vec(vec, h)
    vec_ip1 = circshift(vec, -1)
    vec_im1 = circshift(vec, 1)
    return (vec_ip1 - vec_im1) / (2 * h)
end


function m_T_func(x)
    m_T_func_unnorm = (x) -> exp(-(x - 0.5)^2 / 0.1^2)
    oneOverC, error = quadgk(m_T_func_unnorm, 0, 1) #Why 0, 1? Double check
    return (1 / oneOverC) * m_T_func_unnorm(x)
end

function Gh_func(M)
    return zeros(length(M))
end

#= function conv_fft(M, x_grid, h, δ) # Double check
    N_h = length(M)
    L = x_grid[end] - x_grid[1]   # domain length
    # Force L=1 since "physical" domain is (0,1), extension to (-1,2) is to avoid boundary artifacts? Tried but gives large
    # grid differences, wrapped into (-L/2, L/2]
    raw_diffs = collect(0:N_h-1) .* h
    diffs = raw_diffs .- round.(raw_diffs ./ L) .* L

    # Gaussian kernel on differences
    ϕ_prenormalization = exp.(-(diffs .^ 2) ./ (2δ^2))
    ϕ = ϕ_prenormalization ./ (h * sum(ϕ_prenormalization))   # normalization

    # FFT-based circular convolution
    conv = real(ifft(fft(ϕ) .* fft(M)))
    return conv
end =#

function old_conv_func(m_vec, x_vec, δ)
    ### Double check if this is actually correct, not completely sure...
    ### The "physical domain" is [0,1), but we extend to (-1,2) to avoid boundary artifacts...
    ### What is then the correct way to handle the convolution?
    old_ϕ_δ = (x, δ) -> 1 / (δ * sqrt(2 * π)) * exp(-x^2 / (2 * δ^2))
    N_h = length(m_vec)
    h = x_vec[2] - x_vec[1]
    conv_vec = Vector{Float64}(undef, N_h)

    # Base offsets from reference node and minimum-image wrap
    # Δs = x_vec .- x_vec[1]
    # Δs .-= round.(Δs)  # map to (-0.5, 0.5]
    Δs = (x_vec .- x_vec[1]) .- round.(x_vec .- x_vec[1])  # (-0.5, 0.5] 

    # Periodic kernel weights and discrete normalization
    weights = old_ϕ_δ.(Δs, δ)
    Z = h * sum(weights)
    weights ./= Z

    # Circular convolution with circulant weights (O(N^2), robust and simple)
    for j in 1:N_h
        s = 0.0
        for i in 1:N_h
            idx = mod(j - i, N_h) + 1  # 1-based index
            s += weights[idx] * m_vec[i]
        end
        conv_vec[j] = h * s
    end
    return conv_vec
end



function Fh_func(M_n, t_n, δ, x_grid, h)
    #ϕ_δ=1/(δ* sqrt(2*π)) * exp.(- (x_grid .^ 2) / (2*δ^2))
    f_term_func = (x, t) -> 5 * (x - 0.5 * (1 - sin(2 * π * t)))^2
    conv_term = old_conv_func(M_n, x_grid, δ)
    #conv_term = conv_fft(M_n, x_grid, h, δ)
    f_term = f_term_func.(x_grid, t_n)
    return f_term + conv_term #Double check
end

function H_func(p)
    return p^2 / 2
end

function create_mathcalF(x, U_n, PDL_matrix, M_n, Δt, n, h, ν, x_grid)
    F_hM_n = Fh_func(M_n, (n - 1) * Δt, δ, x_grid, h) #time was T-nΔt in old code??
    δ_hx = central_diff_vec(x, h)
    Hδ_hx = H_func.(δ_hx)
    return x - U_n + Δt * (ν * PDL_matrix * x + Hδ_hx - F_hM_n)
end

##########################

# Matrices 

function create_Jacobian(N_h, U, PDL_matrix, Δt, h, ν) # When Hamiltonian H = p^2 / 2

    δ_hU = central_diff_vec(U, h)

    subdiag = -1 * circshift(δ_hU, -1)[1:N_h-1]
    supdiag = δ_hU[1:N_h-1]

    deriv_Ham_δ_hU_mat = zeros(N_h, N_h) + Tridiagonal(subdiag, zeros(N_h), supdiag)
    deriv_Ham_δ_hU_mat[1, end] = -1 * δ_hU[1]
    deriv_Ham_δ_hU_mat[end, 1] = δ_hU[end]

    J_F = I + Δt * (ν * PDL_matrix + (1 / (2h)) * deriv_Ham_δ_hU_mat)
    return J_F
end


function create_PDL_matrix(N_h, α, h, R) # new method, spectral method. Verified agains old method, and gives zero row sums.
    k = collect(0:N_h-1)
    λ = (2 .- 2 .* cos.(2π .* k ./ N_h)) .^ (α / 2) ./ h^α  # λ_0 = 0
    # First column of the circulant is the inverse FFT of eigenvalues
    c = real(ifft(λ))                       # first column
    D = zeros(Float64, N_h, N_h)
    for j in 1:N_h
        D[:, j] = circshift(c, j - 1)
    end
    return D
end

#######################

# HJB algorithm

function new_HJB_step(num_iter_HJB, N_h, U_n, M_n, PDL_matrix, Δt, h, ν, n, x_grid)
    x_k = U_n

    #println("Starting Netwon's method...")
    #if n > 69
    # println("Inside")
    # @infiltrate
    #end

    #Newtons_progress = Progress(num_it_MFG, desc="Newtons iterations")
    for k in 1:num_iter_HJB
        #println("k=",k) 
        J_F = create_Jacobian(N_h, x_k, PDL_matrix, Δt, h, ν)
        mathcalFx_k = create_mathcalF(x_k, U_n, PDL_matrix, M_n, Δt, n, h, ν, x_grid)
        δ_k = J_F \ mathcalFx_k
        x_kp1 = x_k - δ_k
        x_k = x_kp1
        #next!(Newtons_progress)

        #if n > 69
        #@infiltrate
        # end
    end
    U_np1 = x_k
    return U_np1
end

function new_HJB_solve(N_h, N_T, M_mat, PDL_matrix, Δt, h, ν, num_iter_HJB, x_grid)
    U_mat = Array{Float64}(undef, N_h, N_T)
    U_mat[:, 1] = Gh_func(M_mat[:, 1])

    #println("Starting HJB_solve ...")
    #HJB_progress = Progress(num_it_MFG, desc="HJB iterations")
    for n in 1:(N_T-1)
        #println("n=",n)
        U_mat[:, n+1] = new_HJB_step(num_iter_HJB, N_h, U_mat[:, n], M_mat[:, n], PDL_matrix, Δt, h, ν, n, x_grid)
        #next!(HJB_progress)
    end
    #@infiltrate
    return U_mat
end

##########################

# FPK algorithm

function new_FPK_step(N_h, M_np1, U_np1, PDL_matrix, h, ν, Δt)
    U_diff_vec = central_diff_vec(U_np1, h)
    T = zeros(N_h, N_h) + Tridiagonal(-1 * U_diff_vec[1:(N_h-1)], zeros(N_h), U_diff_vec[2:N_h])
    T[1, end] = -U_diff_vec[end]
    T[end, 1] = U_diff_vec[1]

    tot_mat = (1.0 * I(N_h) + Δt * (ν * PDL_matrix - (1 / (2h)) * T))
    M_n = tot_mat \ M_np1
    #@infiltrate
    return M_n
end

function new_FPK_solve(U_mat, M_T, PDL_matrix, N_h, Δt, N_T, h, ν, α)
    M_mat = Array{Float64}(undef, N_h, N_T)
    M_mat[:, end] = M_T

    #FPK_progress = Progress(num_it_MFG, desc="FPK iterations")
    for j in 1:(N_T-1)
        M_mat[:, N_T-j] = new_FPK_step(N_h, M_mat[:, N_T-j+1], U_mat[:, N_T-j+1], PDL_matrix, h, ν, Δt)
        #next!(FPK_progress)
    end
    return M_mat
end

###############################

# MFG algorithm



function MFG_solve(N_h, N_T, h, Δt, num_it_MFG, num_it_HJB, M_T, α, δ, R, x_grid)
    M_mat = repeat(M_T, 1, N_T)
    U_mat = Array{Float64}(undef, N_h, N_T)
    PDL_matrix = create_PDL_matrix(N_h, α, h, R)

    println("Starting MFG_solve for loop...")
    for j in ProgressBar(1:num_it_MFG)
        #println("MFG iterations. j=", j, "/", num_it_MFG)
        U_mat = new_HJB_solve(N_h, N_T, M_mat, PDL_matrix, Δt, h, ν, num_it_HJB, x_grid)
        M_mat = new_FPK_solve(U_mat, M_T, PDL_matrix, N_h, Δt, N_T, h, ν, α)
        #@infiltrate
    end
    return (U_mat, M_mat)
end





# Run SIMULATION

using DelimitedFiles;
using Glob
using JSON

println("RUN SIMULATION")
####Choose grid sizes and parameters:
h_reference = 1 / 2^9
h_list = [1 / 2^7, 1 / 2^8, h_reference]

α = 1.5
x_l = -1
x_r = 2
Δt = 0.001
t_0 = 0
T = 1
ν = 0.09^2
num_it_MFG = 50
num_it_HJB = 20
δ = 0.4
R = 30

###################################+

#### For saving runs ####
runs_folder = "notebooks/new_MFG_convergence_rate_runs/" #Folder to save runs too. NB: Change run number so old runs aren't overwritten.
isdir(runs_folder) || mkpath(runs_folder)

run_files = readdir(runs_folder)
# Extract run numbers
run_numbers = Int[]
for f in run_files
    if occursin(r"^run\d+_", f)  # filenames like "run7_U_mat_..."
        m = match(r"^run(\d+)_", f)
        if m !== nothing
            push!(run_numbers, parse(Int, m.captures[1]))
        end
    end
end

run_number = isempty(run_numbers) ? 1 : maximum(run_numbers) + 1
println("New run_number = $run_number")
##########

params = Dict(
    "h_list" => Float64[],
    "α" => α,
    "x_l" => x_l,
    "x_r" => x_r,
    "Δt" => Δt,
    "t_0" => t_0,
    "T" => T,
    "ν" => ν,
    "num_it_MFG" => num_it_MFG,
    "num_it_HJB" => num_it_HJB,
    "δ" => δ,
    "R" => R,
)

params_file = joinpath(runs_folder, "run$(run_number)_params.json")
###################################


M_list = Array{Float64}[]
U_list = Array{Float64}[]


for h in h_list
    println("")
    println("start for loop, h=", h)
    println("Running MFG_solve with h=", h)

    x_grid = x_l:h:(x_r-h)
    t_vec = t_0:Δt:(T-Δt)
    N_h = length(x_grid)
    N_T = length(t_vec)
    M_T = m_T_func.(x_grid)

    (U_mat, M_mat) = MFG_solve(N_h, N_T, h, Δt, num_it_MFG, num_it_HJB, M_T, α, δ, R, x_grid)

    M_mat = reverse(M_mat, dims=2)
    U_mat = reverse(U_mat, dims=2)
    println("Done running MFG_sovle with h=", h)

    println("size(M_mat): ", size(M_mat))
    println("size(U_mat): ", size(U_mat))

    push!(M_list, M_mat)
    push!(U_list, U_mat)

    println("Writing results to file...")
    writedlm(joinpath(runs_folder, "run$(run_number)_U_mat_conv_h$(h)_deltat$(Δt).csv"), U_mat, ",")
    writedlm(joinpath(runs_folder, "run$(run_number)_M_mat_conv_h$(h)_deltat$(Δt).csv"), M_mat, ",")

    # Add completed h to params and save again
    push!(params["h_list"], h)
    open(params_file, "w") do io
        JSON.print(io, params)
    end
end