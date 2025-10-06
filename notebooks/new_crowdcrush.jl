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


# Functions

function central_diff_vec(vec, h)
    vec_ip1 = circshift(vec, -1)
    vec_im1 = circshift(vec, 1)
    return (vec_ip1 - vec_im1) / (2 * h)
end


function m_T_func(x)
    m_T_func_unnorm = (x) -> exp(-(x - 0.5)^2 / (0.1)^2)
    oneOverC, error = quadgk(m_T_func_unnorm, 0, 1) #Why 0, 1? Double check
    return (1 / oneOverC) * m_T_func_unnorm(x)
end

#= function decreasing_bump_func(x)
    g = (x) -> x < 0 ? 0 : exp(-1 / x)
    h = (x) -> g(x) / (g(x) + g(1 - x))
    j = (x) -> h(0.9 + 20 * x) - h(0.1 + 0.9x)
    return j(x)
end

function Q(x_vec, δ)
    σ = 0.03
    c = 0.97
    Q_1 = ((x) -> 1 / σ * exp(-(x - c)^2 / (4 * σ^2)) + 1 / σ * exp(-(x - c + 1)^2 / (4 * σ^2))).(x_vec)

    Q_2 = 10 * decreasing_bump_func.(x_vec)

    return Q_1 + Q_2
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

function decreasing_bump_func(x)
    g = (x) -> x < 0 ? 0 : exp(-1 / x)
    h = (x) -> g(x) / (g(x) + g(1 - x))
    #j = (x) -> 3 * (h(0.6 - 0.7 * x) + h(-2 + 3 * x))
    j = (x) -> 3 * (h(0.65 - 0.6 * x) + h(-2.3 + 3 * x))
    return j(x)
end

function Q(x_vec, δ)
    Q_2 = decreasing_bump_func.(x_vec)
    return Q_2
end
#= 
function B(M, t_n) # congestion term
    B_ = 0
    cutt = 50

    M < cutt ? B_ = 1 * exp(0.5 * M) : B_ = 1 * exp.(0.5 * cutt)
    #M < cutt ? B_ = 1 * exp(0.5 * M) : B_ = 1 * exp.(0.5 * cutt)
    #M < cutt ? B_ = (1 / 4) * M^2 : B_ = (1 / 4) * cutt^2


    return B_
end =#

function B(M, x_grid, t_n) # congestion term
    #B_ = 0
    B_ = zeros(length(M))
    cutt = 50

    conv_term = old_conv_func(M, x_grid, 0.05)
    for (i, elem) in enumerate(conv_term)
        if elem < cutt
            B_[i] = 1 * exp(0.5 .* elem)
        else
            B_[i] = 1 * exp(0.5 * cutt)
        end
    end

    #M < cutt ? B_ = 1 * exp(0.5 * M) : B_ = 1 * exp.(0.5 * cutt)
    #M < cutt ? B_ = (1 / 4) * M^2 : B_ = (1 / 4) * cutt^2
    return B_, conv_term
end

function Fh_func(M_n, t_n, δ, x_grid)
    C_Q = 1
    C_B = 1
    Q_term = C_Q * Q(x_grid, δ)
    B_term, _ = C_B .* B(M_n, x_grid, 2 - t_n) #B.(M_n, 2 - t_n)

    return Q_term + B_term
end

function Gh_func(M, x_grid, t_n, δ)
    C_G = 1 / 2
    return C_G * Fh_func(M, t_n, δ, x_grid)
end

function H_func(p, C_H)
    return C_H * p^2
end

function create_mathcalF(x, U_n, PDL_matrix, M_n, Δt, n, h, ν, x_grid, δ, δ_hU, C_H)
    # F_hM_n = Fh_func(M_n, (n - 1) * Δt, δ, x_grid, h) #time was T-nΔt in old code??
    #δ_hx = central_diff_vec(x, h)
    Hδ_hx = H_func.(δ_hU, C_H)
    return x - U_n + Δt * (ν * PDL_matrix * x + Hδ_hx) # - F_hM_n)
end

##########################

# Matrices 

function create_Jacobian(N_h, U, PDL_matrix, Δt, h, ν, δ_hU) # When Hamiltonian H = p^2 / 2
    subdiag = -1 * δ_hU[2:N_h]
    supdiag = δ_hU[1:N_h-1]

    deriv_Ham_δ_hU_mat = zeros(N_h, N_h) + Tridiagonal(subdiag, zeros(N_h), supdiag)
    deriv_Ham_δ_hU_mat[1, end] = -1 * δ_hU[1]
    deriv_Ham_δ_hU_mat[end, 1] = δ_hU[end]

    J_F = 1.0 * I(N_h) + Δt * (ν * PDL_matrix + (1 / (2h)) * deriv_Ham_δ_hU_mat)
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
use_gpu = true

function new_HJB_step(num_iter_HJB, N_h, U_n, M_n, PDL_matrix, Δt, h, ν, n, x_grid, δ, C_H, tol=1e-13)
    x_k = U_n

    newton_time = time()
    jacobian_tot_time = 0.0
    mathcalF_tot_time = 0.0
    inverse_tot_time = 0.0

    if use_gpu
        J_F_gpu = CuArray{Float64}(undef, N_h, N_h)
        mathcalF_gpu = CuArray{Float64}(undef, N_h)
    end

    F_hM_n = Fh_func(M_n, (n - 1) * Δt, δ, x_grid) #time was T-nΔt in old code??

    final_k = 0
    final_norm = 0
    for k in 1:num_iter_HJB
        final_k += 1
        #println("k=", k)
        δ_hU = central_diff_vec(x_k, h)

        J_time = time()
        J_F = create_Jacobian(N_h, x_k, PDL_matrix, Δt, h, ν, δ_hU)
        jacobian_tot_time += time() - J_time

        mathcalF_time = time()
        mathcalFx_k = create_mathcalF(x_k, U_n, PDL_matrix, M_n, Δt, n, h, ν, x_grid, δ, δ_hU, C_H) - Δt * F_hM_n
        mathcalF_tot_time += time() - mathcalF_time

        inverse_time = time()

        if use_gpu
            copy!(J_F_gpu, J_F)
            copy!(mathcalF_gpu, mathcalFx_k)
            δ_k = J_F_gpu \ mathcalF_gpu
            δ_k = Array(δ_k)
        else
            δ_k = J_F \ mathcalFx_k
        end

        inverse_tot_time += time() - inverse_time

        #println("norm δ_k: ", norm(δ_k))

        x_kp1 = x_k - δ_k
        x_k = x_kp1

        δ_k_norm = norm(δ_k)
        final_norm = δ_k_norm
        if δ_k_norm < tol
            break
        end
        #next!(Newtons_progress)

        #if n > 69
        #@infiltrate
        # end
    end

    newton_time = time() - newton_time

    time_percentages = [newton_time, jacobian_tot_time / newton_time, mathcalF_tot_time / newton_time, inverse_tot_time / newton_time]

    U_np1 = x_k
    return U_np1, time_percentages, final_k, final_norm
end


#######
function new_HJB_solve(N_h, N_T, M_mat, PDL_matrix, Δt, h, ν, num_iter_HJB, x_grid, δ, C_H)
    U_mat = Array{Float64}(undef, N_h, N_T)
    U_mat[:, 1] = Gh_func(M_mat[:, 1], x_grid, 0, δ)

    #println("Starting HJB_solve ...")
    #HJB_progress = Progress(num_it_MFG, desc="HJB iterations")

    method = ""
    avg_times = [0.0, 0.0, 0.0, 0.0]
    newton_iterations = Float64[]
    newton_final_norms = Float64[]
    hjb_solve_time = time()
    for n in 1:(N_T-1)
        #println("n=",n)

        U_mat[:, n+1], time_percentages, final_k, final_norm = new_HJB_step(num_iter_HJB, N_h, U_mat[:, n], M_mat[:, n], PDL_matrix, Δt, h, ν, n, x_grid, δ, C_H)
        #println("n: ", n, "Time and time percentages [N (Newtons), J/N, F/N, inverse/N]: ", time_percentages)
        push!(newton_iterations, final_k)
        push!(newton_final_norms, final_norm)
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
    println("Newton iterations, min, max, avg: ", minimum(newton_iterations), ". ", maximum(newton_iterations), ". ", sum(newton_iterations) / length(newton_iterations))
    println("Newton final norms, min, max, avg: ", minimum(newton_final_norms), ". ", maximum(newton_final_norms), ". ", sum(newton_final_norms) / length(newton_final_norms))
    #@infiltrate
    return U_mat
end

##########################

# FPK algorithm

function new_FPK_step(N_h, M_np1, U_np1, PDL_matrix, h, ν, Δt, C_H)
    U_diff_vec = central_diff_vec(U_np1, h)
    T = zeros(N_h, N_h) + Tridiagonal(-1 * U_diff_vec[1:(N_h-1)], zeros(N_h), U_diff_vec[2:N_h])
    T[1, end] = -U_diff_vec[end]
    T[end, 1] = U_diff_vec[1]

    tot_mat = (1.0 * I(N_h) + Δt * (ν * PDL_matrix - (2 * C_H) * (1 / (2h)) * T))

    if use_gpu
        M_n = CuArray(tot_mat) \ CuArray(M_np1)
        M_n = Array(M_n)
    else
        M_n = tot_mat \ M_np1
    end
    #@infiltrate
    return M_n
end

function new_FPK_solve(U_mat, M_T, PDL_matrix, N_h, Δt, N_T, h, ν, α, C_H)
    M_mat = Array{Float64}(undef, N_h, N_T)
    M_mat[:, end] = M_T

    #FPK_progress = Progress(num_it_MFG, desc="FPK iterations")
    fpk_avg_time = 0.0
    fpk_solve_time = time()
    for j in 1:(N_T-1)
        fpk_time = time()
        M_mat[:, N_T-j] = new_FPK_step(N_h, M_mat[:, N_T-j+1], U_mat[:, N_T-j+1], PDL_matrix, h, ν, Δt, C_H)
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



function MFG_solve(N_h, N_T, h, Δt, num_it_MFG, num_it_HJB, M_T, α, δ, R, x_grid, C_H, write_iteration_results=false)
    M_mat = repeat(Float64.(M_T), 1, N_T)
    #U_mat = Array{Float64}(undef, N_h, N_T)
    U_mat = zeros(Float64, N_h, N_T)
    PDL_matrix = create_PDL_matrix(N_h, α, h, R)

    U_mat_temp = zeros(Float64, N_h, N_T)
    M_mat_temp = zeros(Float64, N_h, N_T)

    println("Starting MFG_solve for loop...")

    for j in 1:num_it_MFG #ProgressBar(1:num_it_MFG)
        mfg_solve_time = time()
        println("j: ", j)

        U_mat_temp = new_HJB_solve(N_h, N_T, M_mat, PDL_matrix, Δt, h, ν, num_it_HJB, x_grid, δ, C_H)

        M_mat_temp = new_FPK_solve(U_mat_temp, M_T, PDL_matrix, N_h, Δt, N_T, h, ν, α, C_H) #Use U_mat or U_mat_temp?


        U_error_start = max_error(U_mat[0 .<= x_grid .< 1, 1], U_mat_temp[0 .<= x_grid .< 1, 1])
        U_error_end = max_error(U_mat[0 .<= x_grid .< 1, end], U_mat_temp[0 .<= x_grid .< 1, end])
        M_error_start = max_error(M_mat[0 .<= x_grid .< 1, 1], M_mat_temp[0 .<= x_grid .< 1, 1])
        M_error_end = max_error(M_mat[0 .<= x_grid .< 1, end], M_mat_temp[0 .<= x_grid .< 1, end])
        M_l1_error_start = l1_error(M_mat[0 .<= x_grid .< 1, 1], M_mat_temp[0 .<= x_grid .< 1, 1])
        M_l1_error_end = l1_error(M_mat[0 .<= x_grid .< 1, end], M_mat_temp[0 .<= x_grid .< 1, end])


        if write_iteration_results
            if j == 1
                println("Writing iteration number 0, the initial guess...")
                println("Writing U: ", "run$(run_number)_iter0_U_mat_conv_h$(h)_deltat$(Δt).csv")
                writedlm(joinpath(iterations_folder, "run$(run_number)_iter0_U_mat_conv_h$(h)_deltat$(Δt).csv"), reverse(U_mat, dims=2), ",")
                println("Writing M: ", "run$(run_number)_iter0_M_mat_conv_h$(h)_deltat$(Δt).csv")
                writedlm(joinpath(iterations_folder, "run$(run_number)_iter0_M_mat_conv_h$(h)_deltat$(Δt).csv"), reverse(M_mat, dims=2), ",")
                println("Writing iteration completed.")
            end

            println("Writing iteration number ", j, "...")
            println("Writing U: ", "run$(run_number)_iter$(j)_U_mat_conv_h$(h)_deltat$(Δt).csv")
            writedlm(joinpath(iterations_folder, "run$(run_number)_iter$(j)_U_mat_conv_h$(h)_deltat$(Δt).csv"), reverse(U_mat_temp, dims=2), ",")
            println("Writing M: ", "run$(run_number)_iter$(j)_M_mat_conv_h$(h)_deltat$(Δt).csv")
            writedlm(joinpath(iterations_folder, "run$(run_number)_iter$(j)_M_mat_conv_h$(h)_deltat$(Δt).csv"), reverse(M_mat_temp, dims=2), ",")
            println("Writing iteration completed.")
        end

        println("errors:")
        println("U_error_start: ", U_error_start)
        println("U_error_end: ", U_error_end)
        println("M_error_start: ", M_error_start)
        println("M_error_end: ", M_error_end)
        println("M_l1_error_start: ", M_l1_error_start)
        println("M_l1_error_end: ", M_l1_error_end)

        U_mat = U_mat_temp
        M_mat = M_mat_temp

        #@infiltrate
        println("j: ", j, ". MFG solve time: ", time() - mfg_solve_time)
        println("!!!!!!!!!!")
        println("")
        if all([
            U_error_start < 1e-10,
            U_error_end < 1e-10,
            M_error_start < 1e-10,
            M_error_end < 1e-10,
            M_l1_error_start < 1e-10,
            M_l1_error_end < 1e-10
        ])
            println("Below tolerance. Breaking.")
            break
        end
    end
    return (U_mat, M_mat)
end



function max_error(vec, wrt_vec, debug=false)
    #Assuming boundary 
    max_difference = maximum(abs.(vec - wrt_vec))
    max_index_diff = argmax(abs.(vec - wrt_vec))
    max_index_wrt_vec = argmax(abs.(wrt_vec))
    if debug
        println("max_index_diff: ", max_index_diff, ". Total length: ", length(vec), ". x-pos: ", string((max_index_diff / length(vec))))
        println("max_index_wrt_vec: ", max_index_wrt_vec, ". Total length: ", length(vec), ". x-pos: ", string((max_index_wrt_vec / length(vec))))
    end
    return max_difference
end

function l1_error(vec, wrt_vec, debug=false)
    if debug
        println("length(vec): ", length(vec))
        println("length(wrt_vec): ", length(wrt_vec))
    end
    l1 = sum(abs.(vec - wrt_vec)) #*h
    return l1
end

# Run SIMULATION

using DelimitedFiles;
using Glob
using JSON

println("RUN SIMULATION")
####Choose grid sizes and parameters:
h_reference = 1 / 2^11
h_list = [1 / 2^8]

α = 1.5
x_l = -1
x_r = 2
Δt = 0.005
t_0 = 0
T = 2
ν = 0.1
num_it_MFG = 50
num_it_HJB = 50
δ = 0.05
R = 30

C_L = 10
C_H = 1 / C_L

###################################+

#### For saving runs ####
runs_folder = "notebooks/new_crowdcrush_runs/" #Folder to save runs too. NB: Change run number so old runs aren't overwritten.
iterations_folder = "notebooks/new_crowdcrush_iteration_plots/"
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
    "C_H" => C_H,
)

params_file = joinpath(runs_folder, "run$(run_number)_params.json")
###################################


M_list = Array{Float64}[]
U_list = Array{Float64}[]

println("Use GPU to solve linear systems: ", use_gpu)

write_iters = true

for h in h_list
    println("")
    println("start for loop, h=", h)
    println("Running MFG_solve with h=", h)

    x_grid = x_l:h:(x_r-h)
    t_vec = t_0:Δt:(T-Δt)
    N_h = length(x_grid)
    N_T = length(t_vec)
    M_T = m_T_func.(x_grid)

    #if write_iters
    push!(params["h_list"], h)
    open(params_file, "w") do io
        JSON.print(io, params)
    end
    #end
    (U_mat, M_mat) = MFG_solve(N_h, N_T, h, Δt, num_it_MFG, num_it_HJB, M_T, α, δ, R, x_grid, C_H, write_iters)


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
    #=     if !write_iters
            push!(params["h_list"], h)
            open(params_file, "w") do io
                JSON.print(io, params)
            end
        end =#
end