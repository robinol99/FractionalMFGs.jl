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
using Krylov
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

function central_diff_vec!(cent_diff_gpu, vec, h)
    vec_ip1 = circshift(vec, -1)
    vec_im1 = circshift(vec, 1)
    @. cent_diff_gpu = (vec_ip1 - vec_im1) / (2 * h)
    return cent_diff_gpu
end

function create_mathcalF!(
    F_gpu, x_gpu, U_n_gpu, PDL_matrix_gpu, Δt, h, ν, cent_diff_vec, Hδ_hx
)
    mul!(F_gpu, PDL_matrix_gpu, x_gpu)
    @. F_gpu .*= Δt * ν

    Hδ_hx .= H_func.(cent_diff_vec)
    F_gpu .+= Δt .* Hδ_hx

    @. F_gpu .+= x_gpu - U_n_gpu

    return F_gpu
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


function create_Jacobian!(J_F_gpu, N_h, δ_hU_gpu, PDL_matrix_gpu, Δt, h, ν)
    # Reset J_F_gpu
    fill!(J_F_gpu, 0.0)

    # Interior tridiagonal
    for i in 2:N_h-1
        @inbounds J_F_gpu[i, i-1] = -Δt * δ_hU_gpu[i] / (2h)
        @inbounds J_F_gpu[i, i] = 1.0
        @inbounds J_F_gpu[i, i+1] = Δt * δ_hU_gpu[i] / (2h)
    end

    # Handle periodic boundaries
    J_F_gpu[1, 1] = 1.0
    J_F_gpu[1, 2] = Δt * δ_hU_gpu[1] / (2h)
    J_F_gpu[1, end] = -Δt * δ_hU_gpu[1] / (2h)
    J_F_gpu[end, end] = 1.0
    J_F_gpu[end, end-1] = -Δt * δ_hU_gpu[end] / (2h)
    J_F_gpu[end, 1] = Δt * δ_hU_gpu[end] / (2h)

    # Add PDL contribution
    J_F_gpu .+= Δt * ν * PDL_matrix_gpu

    #=     # Precompute coefficients
        coef = Δt / (2h)

        # Interior indices
        interior = 2:N_h-1

        # Fill diagonals (broadcast over slices)
        @views J_F_gpu[interior, interior] .= 1.0                           # main diagonal
        @views J_F_gpu[interior, interior.-1] .= .-coef .* δ_hU_gpu[interior]  # sub-diagonal
        @views J_F_gpu[interior, interior.+1] .= coef .* δ_hU_gpu[interior]   # super-diagonal

        # Periodic boundaries — do explicitly once per edge (2 writes per edge is fine)
        @views begin
            J_F_gpu[1, 1] = 1.0
            J_F_gpu[1, 2] = coef * δ_hU_gpu[1]
            J_F_gpu[1, end] = -coef * δ_hU_gpu[1]

            J_F_gpu[end, end] = 1.0
            J_F_gpu[end, end-1] = -coef * δ_hU_gpu[end]
            J_F_gpu[end, 1] = coef * δ_hU_gpu[end]
        end

        # Add PDL contribution (one broadcasted kernel)
        J_F_gpu .+= Δt * ν * PDL_matrix_gpu =#


    return J_F_gpu
end



#######################

# HJB algorithm
use_gpu = true
CUDA.allowscalar(true)

function new_HJB_step!(num_iter_HJB, N_h, U_np1_gpu, U_n_gpu, M_n_gpu, PDL_matrix_gpu, Δt, h, ν, n, x_grid, δ, J_F_gpu, mathcalF_gpu, δ_k_gpu, cent_diff_vec, Hδ_hx, tol=1e-10)
    @. U_np1_gpu = U_n_gpu

    newton_time = time()
    jacobian_tot_time = 0.0
    mathcalF_tot_time = 0.0
    inverse_tot_time = 0.0


    F_hM_n = CuArray(Fh_func(Array(M_n_gpu), (n - 1) * Δt, δ, x_grid, h)) #time was T-nΔt in old code??

    for k in 1:num_iter_HJB

        central_diff_vec!(cent_diff_vec, U_np1_gpu, h)

        mathcalF_time = time()
        create_mathcalF!(mathcalF_gpu, U_np1_gpu, U_n_gpu, PDL_matrix_gpu, Δt, h, ν, cent_diff_vec, Hδ_hx)
        mathcalF_gpu .-= Δt .* F_hM_n
        mathcalF_tot_time += time() - mathcalF_time


        J_time = time()
        create_Jacobian!(J_F_gpu, N_h, cent_diff_vec, PDL_matrix_gpu, Δt, h, ν)
        jacobian_tot_time += time() - J_time


        inverse_time = time()



        # solve with GMRES (good general choice)
        δ_k_gpu .= gmres(J_F_gpu, mathcalF_gpu; rtol=1e-10, itmax=500)[1]

        #=         δ_k_tmp, stats = gmres(J_F_gpu, mathcalF_gpu; tol=1e-10, maxiter=500)
                δ_k_gpu .= δ_k_tmp


                if !stats.converged
                    @warn "GMRES did not converge, residual = $(stats.resnorm)"
                end =#

        inverse_tot_time += time() - inverse_time

        @. U_np1_gpu = U_np1_gpu - δ_k_gpu

        δ_k_norm = norm(δ_k_gpu)
        if δ_k_norm < tol
            break
        end
    end

    newton_time = time() - newton_time
    time_percentages = [newton_time, jacobian_tot_time / newton_time, mathcalF_tot_time / newton_time, inverse_tot_time / newton_time]

    return time_percentages
end


#######
function new_HJB_solve(N_h, N_T, M_mat, PDL_matrix, Δt, h, ν, num_iter_HJB, x_grid, δ)
    U_mat = Array{Float64}(undef, N_h, N_T)
    U_mat[:, 1] = Gh_func(M_mat[:, 1])

    #println("Starting HJB_solve ...")
    #HJB_progress = Progress(num_it_MFG, desc="HJB iterations")

    method = ""
    avg_times = [0.0, 0.0, 0.0, 0.0]

    hjb_solve_time = time()


    U_mat_gpu = CuArray(U_mat)
    M_mat_gpu = CuArray(M_mat)
    PDL_matrix_gpu = CuArray(PDL_matrix)
    J_F_gpu = CuArray{Float64}(undef, N_h, N_h)
    mathcalF_gpu = CuArray{Float64}(undef, N_h)
    δ_k_gpu = CuArray{Float64}(undef, N_h)
    cent_diff_vec = CuArray{Float64}(undef, N_h)
    Hδ_hx = CuArray{Float64}(undef, N_h)

    for n in 1:(N_T-1)
        #println("n=",n)
        time_percentages = new_HJB_step!(num_iter_HJB, N_h, view(U_mat_gpu, :, n + 1), view(U_mat_gpu, :, n), view(M_mat_gpu, :, n), PDL_matrix_gpu, Δt, h, ν, n, x_grid, δ, J_F_gpu, mathcalF_gpu, δ_k_gpu, cent_diff_vec, Hδ_hx)
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

    U_mat = Array(U_mat_gpu)
    return U_mat
end

##########################

# FPK algorithm

function new_FPK_step!(N_h, M_n, M_np1, U_np1, PDL_matrix, h, ν, Δt)
    U_diff_vec = central_diff_vec(Array(U_np1), h)
    T = zeros(N_h, N_h) + Tridiagonal(-1 * U_diff_vec[1:(N_h-1)], zeros(N_h), U_diff_vec[2:N_h])
    T[1, end] = -U_diff_vec[end]
    T[end, 1] = U_diff_vec[1]

    tot_mat = CuArray((1.0 * I(N_h) + Δt * (ν * PDL_matrix - (1 / (2h)) * T)))

    #δ_k_gpu .= gmres(J_F_gpu, mathcalF_gpu; rtol=1e-10, itmax=500)[1]

    @. M_n = gmres(tot_mat, M_np1; rtol=1e-10, itmax=500)[1]
    #@infiltrate
    return
end

function new_FPK_solve(U_mat, M_T, PDL_matrix, N_h, Δt, N_T, h, ν, α)
    M_mat = Array{Float64}(undef, N_h, N_T)
    M_mat[:, end] = M_T

    U_mat_gpu = CuArray(U_mat)
    M_mat_gpu = CuArray(M_mat)

    #FPK_progress = Progress(num_it_MFG, desc="FPK iterations")
    fpk_avg_time = 0.0
    fpk_solve_time = time()
    for j in 1:(N_T-1)
        fpk_time = time()
        #M_mat[:, N_T-j] = new_FPK_step(N_h, M_mat[:, N_T-j+1], U_mat[:, N_T-j+1], PDL_matrix, h, ν, Δt)
        new_FPK_step!(N_h, view(M_mat_gpu, :, N_T - j), view(M_mat_gpu, :, N_T - j + 1), view(U_mat_gpu, :, N_T - j + 1), PDL_matrix, h, ν, Δt)
        fpk_avg_time += time() - fpk_time
        #next!(FPK_progress)
    end
    fpk_avg_time = fpk_avg_time / (N_T - 1)
    fpk_solve_time = time() - fpk_solve_time
    println("FPK solve time: ", fpk_solve_time, ". FPK step avg. time: ", fpk_avg_time)

    M_mat = Array(M_mat_gpu)
    return M_mat
end

###############################

# MFG algorithm
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


function MFG_solve(N_h, N_T, h, Δt, num_it_MFG, num_it_HJB, M_T, α, δ, R, x_grid, write_iteration_results=false)
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
        #println("MFG iterations. j=", j, "/", num_it_MFG)
        U_mat_temp = new_HJB_solve(N_h, N_T, M_mat, PDL_matrix, Δt, h, ν, num_it_HJB, x_grid, δ)
        #@infiltrate
        M_mat_temp = new_FPK_solve(U_mat_temp, M_T, PDL_matrix, N_h, Δt, N_T, h, ν, α) #Use U_mat or U_mat_temp?


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





# Run SIMULATION

using DelimitedFiles;
using Glob
using JSON

println("RUN SIMULATION")
####Choose grid sizes and parameters:
h_reference = 1 / 2^11
h_list = [h_reference] #[1 / 2^7, 1 / 2^8, 1 / 2^9, h_reference]

α = 1.5
x_l = -1
x_r = 2
Δt = 0.0001
t_0 = 0
T = 0.5
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
    if write_iters
        push!(params["h_list"], h)
        open(params_file, "w") do io
            JSON.print(io, params)
        end
    end
    (U_mat, M_mat) = MFG_solve(N_h, N_T, h, Δt, num_it_MFG, num_it_HJB, M_T, α, δ, R, x_grid, write_iters)


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
    if !write_iters
        push!(params["h_list"], h)
        open(params_file, "w") do io
            JSON.print(io, params)
        end
    end
end