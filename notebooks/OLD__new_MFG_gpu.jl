#= using Pkg

# To update all packages in the current environment
Pkg.update() =#

using ProgressBars;
using ProgressMeter
using LinearAlgebra, Plots, FFTW;
using LoggingExtras, ProgressLogging, Logging, TerminalLoggers
using Infiltrator
import LinearSolve as LS
using SparseArrays
using QuadGK;
using SpecialFunctions: gamma, loggamma, zeta
using CUDA
println("BLAS.get_num_threads(): ", BLAS.get_num_threads())
#= BLAS.get_num_threads()
println("BLAS.get_num_threads(): ", BLAS.get_num_threads())
println("Sys.CPU_THREADS: ", Sys.CPU_THREADS)
BLAS.set_num_threads(Sys.CPU_THREADS)
println("BLAS.get_num_threads(): ", BLAS.get_num_threads()) =#

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


##########################

# Matrices 


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


function conv_fft_gpu(M_n_gpu::CuArray{Float64}, x_grid_gpu::CuArray{Float64}, h::Float64, δ::Float64)
    N_h = length(M_n_gpu)

    # Compute periodic kernel ϕ_δ on GPU
    Δs = x_grid_gpu .- x_grid_gpu[1]
    Δs .-= round.(Δs)           # (-0.5,0.5]
    ϕ_δ = @. 1 / (δ * sqrt(2π)) * exp(-Δs^2 / (2 * δ^2))

    # Normalize kernel
    Z = h * sum(ϕ_δ)
    ϕ_δ ./= Z

    # FFT-based convolution
    fft_M = fft(M_n_gpu)
    fft_ϕ = fft(ϕ_δ)
    conv_fft = ifft(fft_M .* fft_ϕ)

    # Multiply by h to match original scaling and take real part
    return h .* real(conv_fft)
end

using GPUArrays
function create_Jacobian_gpu_full!(J_F_gpu, N_h, U_gpu, PDL_matrix_gpu, Δt, h, ν)
    # U_gpu is CuArray{Float64}
    @allowscalar begin
        # Compute central differences on GPU
        δ_hU_gpu = similar(U_gpu)
        δ_hU_gpu[2:N_h-1] .= (U_gpu[3:N_h] .- U_gpu[1:N_h-2]) ./ (2h)

        # First and last entries (periodic BC)
        δ_hU_gpu[1] = (U_gpu[2] - U_gpu[end]) / (2h)
        δ_hU_gpu[end] = (U_gpu[1] - U_gpu[end-1]) / (2h)

        # Fill tridiagonal elements directly on GPU
        # Create a zero matrix
        fill!(J_F_gpu, 0.0)

        subdiag = -δ_hU_gpu[2:end] / (2h)
        supdiag = δ_hU_gpu[1:end-1] / (2h)
        diag = ones(Float64, N_h)  # diagonal ones

        # For periodic BCs, we handle them separately
        J_F_gpu .= 0.0
        # Fill tridiagonal
        for i in 2:N_h-1
            J_F_gpu[i, i] = diag[i]
            J_F_gpu[i, i-1] = subdiag[i-1]
            J_F_gpu[i, i+1] = supdiag[i]
        end
        # Handle first and last row periodic entries
        J_F_gpu[1, 1] = diag[1]
        J_F_gpu[1, 2] = supdiag[1]
        J_F_gpu[1, end] = -δ_hU_gpu[1] / (2h)
        J_F_gpu[end, end] = diag[end]
        J_F_gpu[end, end-1] = subdiag[end-1]
        J_F_gpu[end, 1] = δ_hU_gpu[end]

        # Add PDL contribution scaled by ν * Δt
        J_F_gpu .+= Δt * ν * PDL_matrix_gpu
    end
end

function create_mathcalF_gpu!(mathcalF_gpu, x_gpu, U_n_gpu, PDL_matrix_gpu, M_n_gpu, Δt, n, h, ν, x_grid_gpu, δ)
    # Compute F_hM_n fully on GPU
    @allowscalar begin
        f_term = @. 5 * (x_grid_gpu - 0.5 * (1 - sin(2π * (n - 1) * Δt)))^2
        conv_term = conv_fft_gpu(M_n_gpu, x_grid_gpu, h, δ)
        F_hM_n_gpu = f_term + conv_term

        # Compute derivative term

        δ_hx_gpu = central_diff_vec(x_gpu, h)
    end
    Hδ_hx_gpu = @. δ_hx_gpu^2 / 2

    # Put everything together
    mathcalF_gpu .= x_gpu - U_n_gpu + Δt * (ν * PDL_matrix_gpu * x_gpu + Hδ_hx_gpu - F_hM_n_gpu)
end

function new_HJB_step(num_iter_HJB, N_h, U_n, M_n, PDL_matrix, Δt, h, ν, n, x_grid, δ, tol=1e-10)
    J_F_gpu = CuArray{Float64}(undef, N_h, N_h)
    mathcalF_gpu = CuArray{Float64}(undef, N_h)
    x_grid_gpu = CuArray(x_grid)
    M_n_gpu = CuArray(M_n)
    U_n_gpu = CuArray(U_n)
    x_k_gpu = CuArray(U_n)
    x_kp1_gpu = similar(x_k_gpu)
    δ_k_gpu = similar(x_k_gpu)

    newton_time = time()
    jacobian_tot_time = 0.0
    mathcalF_tot_time = 0.0
    inverse_tot_time = 0.0


    for k in 1:num_iter_HJB
        #println("k=",k) 

        J_time = time()
        create_Jacobian_gpu_full!(J_F_gpu, N_h, x_k_gpu, PDL_matrix, Δt, h, ν)
        jacobian_tot_time += time() - J_time

        mathcalF_time = time()
        create_mathcalF_gpu!(mathcalF_gpu, x_k_gpu, U_n_gpu, PDL_matrix, M_n_gpu, Δt, n, h, ν, x_grid_gpu, δ)
        mathcalF_tot_time += time() - mathcalF_time

        inverse_time = time()

        δ_k_gpu .= J_F_gpu \ mathcalF_gpu

        inverse_tot_time += time() - inverse_time

        #println("norm δ_k: ", norm(δ_k))

        x_kp1_gpu .= x_k_gpu .- δ_k_gpu
        x_k_gpu .= x_kp1_gpu

        if norm(δ_k_gpu) < tol
            break
        end
    end

    newton_time = time() - newton_time

    time_percentages = [newton_time, jacobian_tot_time / newton_time, mathcalF_tot_time / newton_time, inverse_tot_time / newton_time]

    U_np1 = Array(x_k_gpu)
    return U_np1, time_percentages
end

#######

function new_HJB_solve(N_h, N_T, M_mat, PDL_matrix, Δt, h, ν, num_iter_HJB, x_grid, δ)
    U_mat = Array{Float64}(undef, N_h, N_T)
    U_mat[:, 1] = Gh_func(M_mat[:, 1])

    #println("Starting HJB_solve ...")
    #HJB_progress = Progress(num_it_MFG, desc="HJB iterations")

    avg_times = [0.0, 0.0, 0.0, 0.0]

    hjb_solve_time = time()
    for n in 1:(N_T-1)
        #println("n=",n)
        U_mat[:, n+1], time_percentages = new_HJB_step(num_iter_HJB, N_h, U_mat[:, n], M_mat[:, n], PDL_matrix, Δt, h, ν, n, x_grid, δ)
        #println("n: ", n, "Time and time percentages [N (Newtons), J/N, F/N, inverse/N]: ", time_percentages)
        avg_times[1] += time_percentages[1]
        avg_times[2] += time_percentages[2]
        avg_times[3] += time_percentages[3]
        avg_times[4] += time_percentages[4]
        #println("avg_time: ", avg_times)
        #next!(HJB_progress)
    end

    avg_times[1] = avg_times[1] / (N_T - 1)
    avg_times[2] = avg_times[2] / (N_T - 1)
    avg_times[3] = avg_times[3] / (N_T - 1)
    avg_times[4] = avg_times[4] / (N_T - 1)
    println("HJB solve time: ", time() - hjb_solve_time, ". HJB step avg time and time percentages [N (Newtons), J/N, F/N, inverse/N]: ", avg_times)
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

    if use_gpu
        M_n = CuArray(tot_mat) \ CuArray(M_np1)
        M_n = Array(M_n)
    else
        M_n = tot_mat \ M_np1
    end
    #@infiltrate
    return M_n
end

function new_FPK_solve(U_mat, M_T, PDL_matrix, N_h, Δt, N_T, h, ν, α)
    M_mat = Array{Float64}(undef, N_h, N_T)
    M_mat[:, end] = M_T

    #FPK_progress = Progress(num_it_MFG, desc="FPK iterations")
    fpk_avg_time = 0.0
    fpk_solve_time = time()
    for j in 1:(N_T-1)
        fpk_time = time()
        M_mat[:, N_T-j] = new_FPK_step(N_h, M_mat[:, N_T-j+1], U_mat[:, N_T-j+1], PDL_matrix, h, ν, Δt)
        fpk_avg_time += time() - fpk_time
        #next!(FPK_progress)
    end
    fpk_avg_time = fpk_avg_time / (N_T - 1)
    fpk_solve_time = time() - fpk_solve_time
    println("FPK solve time: ", fpk_solve_time, ". FPK step avg. time: ", fpk_avg_time)
    return M_mat
end

###############################

# MFG algorithm



function MFG_solve(N_h, N_T, h, Δt, num_it_MFG, num_it_HJB, M_T, α, δ, R, x_grid)
    M_mat = repeat(M_T, 1, N_T)
    U_mat = Array{Float64}(undef, N_h, N_T)

    PDL_matrix = create_PDL_matrix(N_h, α, h, R)
    PDL_matrix_gpu = CuArray(PDL_matrix)

    println("Starting MFG_solve for loop...")
    for j in 1:num_it_MFG #ProgressBar(1:num_it_MFG)
        mfg_solve_time = time()
        println("j: ", j)
        #println("MFG iterations. j=", j, "/", num_it_MFG)
        U_mat = new_HJB_solve(N_h, N_T, M_mat, PDL_matrix_gpu, Δt, h, ν, num_it_HJB, x_grid, δ)
        #@infiltrate
        M_mat = new_FPK_solve(U_mat, M_T, PDL_matrix, N_h, Δt, N_T, h, ν, α)
        #@infiltrate
        println("j: ", j, ". MFG solve time: ", time() - mfg_solve_time)
        println("!!!!!!!!!!")
        println("")
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

use_gpu = true
println("Use GPU to solve linear systems: ", use_gpu)

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