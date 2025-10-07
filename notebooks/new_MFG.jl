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


#= 
function extended_conv_kernel(x, c, δ, x_l, x_r)
    f = (y) -> exp(-(y - c)^2 / (2 * δ^2)) # 1 / (δ * sqrt(2 * π)) 
    K = 1000
    L = x_r - x_l
    s = 0.0
    for k in -K:K
        s += f(x + k * L)
    end
    return s
end

function normalized_ext_conv_kernel(x_grid, x_l, x_r, c, δ)
    vals = [extended_conv_kernel(x, c, δ, x_l, x_r) for x in x_grid]
    dx = x_grid[2] - x_grid[1]
    integral = sum(vals) * dx
    return vals ./ integral
end

function fft_conv(x_grid, vec_to_convolve, x_l, x_r, δ)
    c = x_l
    kernel = normalized_ext_conv_kernel(x_grid, x_l, x_r, c, δ)
    Ff = fft(kernel)
    Gg = fft(vec_to_convolve)
    conv_coeffs = Ff .* Gg
    convolution_vec = real(ifft(conv_coeffs))
    return convolution_vec .* (x_r - x_l) / length(kernel)
end

function new_m_T_func(x_grid, c)
    δ = 0.1 / sqrt(2)
    h = x_grid[2] - x_grid[1]
    x_l = x_grid[1]
    x_r = x_grid[end] + h
    return normalized_ext_conv_kernel(x_grid, x_l, x_r, c, δ)
end



function Fh_func(M_n, t_n, δ, x_grid, h)
    f_term_func = (x, t) -> 5 * (x - 0.5 * (1 - sin(2 * π * t)))^2

    #h = x_grid[2] - x_grid[1]
    x_l = x_grid[1]
    x_r = x_grid[end] + h
    conv_term = fft_conv(x_grid, M_n, x_l, x_r, δ)

    f_term = f_term_func.(x_grid, t_n)
    return f_term + conv_term #Double check
end =#

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

function create_mathcalF(x, U_n, PDL_matrix, M_n, Δt, n, h, ν, x_grid, δ, δ_hU)
    # F_hM_n = Fh_func(M_n, (n - 1) * Δt, δ, x_grid, h) #time was T-nΔt in old code??
    #δ_hx = central_diff_vec(x, h)
    Hδ_hx = H_func.(δ_hU)
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

function new_HJB_step(num_iter_HJB, N_h, U_n, M_n, PDL_matrix, Δt, h, ν, n, x_grid, δ, tol=1e-13)
    x_k = U_n

    #println("Starting Netwon's method...")
    #if n > 69
    # println("Inside")
    # @infiltrate
    #end

    #Newtons_progress = Progress(num_it_MFG, desc="Newtons iterations")
    #println("Recording time spent in Newton's for n=", n, "...")
    newton_time = time()
    jacobian_tot_time = 0.0
    mathcalF_tot_time = 0.0
    inverse_tot_time = 0.0

    if use_gpu
        J_F_gpu = CuArray{Float64}(undef, N_h, N_h)
        mathcalF_gpu = CuArray{Float64}(undef, N_h)
    end

    F_hM_n = Fh_func(M_n, (n - 1) * Δt, δ, x_grid, h) #time was T-nΔt in old code??

    final_k = 0
    for k in 1:num_iter_HJB
        final_k += 1
        #println("k=",k) 
        δ_hU = central_diff_vec(x_k, h)

        J_time = time()
        J_F = create_Jacobian(N_h, x_k, PDL_matrix, Δt, h, ν, δ_hU)
        jacobian_tot_time += time() - J_time

        mathcalF_time = time()
        mathcalFx_k = create_mathcalF(x_k, U_n, PDL_matrix, M_n, Δt, n, h, ν, x_grid, δ, δ_hU) - Δt * F_hM_n
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
        if δ_k_norm < tol
            break
        end
        #next!(Newtons_progress)

        #if n > 69
        #@infiltrate
        # end
    end

    newton_time = time() - newton_time

    #println("newtons method. n: ", n)
    #println("newton time: ", newton_time)
    #println("jacobian tot time: ", jacobian_tot_time)
    #println("mathcalF tot time: ", mathcalF_tot_time)
    #println("inverse to time: ", inverse_tot_time)

    time_percentages = [newton_time, jacobian_tot_time / newton_time, mathcalF_tot_time / newton_time, inverse_tot_time / newton_time]

    U_np1 = x_k
    return U_np1, time_percentages, final_k
end


# TRY EXPLICIT:

function explicit_HJB_step(num_iter_HJB, N_h, U_n, M_n, PDL_matrix, Δt, h, ν, n, x_grid)
    F_hM_n = Fh_func(M_n, (n - 1) * Δt, δ, x_grid, h) #time was T-nΔt in old code??
    δ_hU_n = central_diff_vec(U_n, h)
    Hδ_hU_n = H_func.(δ_hU_n)
    PDL_prod = PDL_matrix * U_n
    U_np1 = U_n - Δt * (ν * PDL_prod + Hδ_hU_n - F_hM_n)
    return U_np1
end

# Above didn't work. TRY IMPLICIT-EXPLICIT:

function imp_exp_HJB_step(num_iter_HJB, N_h, U_n, M_n, PDL_matrix, Δt, h, ν, n, x_grid, θ=0.9)
    #implicit half step
    implicit_mat = (I + Δt / 2 * θ * ν * PDL_matrix)
    U_np1over2 = implicit_mat \ U_n

    #explicit half step
    F_hM_n = Fh_func(M_n, (n - 1) * Δt, δ, x_grid, h) #time was T-nΔt in old code??
    δ_hU_np1over2 = central_diff_vec(U_np1over2, h)
    Hδ_hU_np1over2 = H_func.(δ_hU_np1over2)
    PDL_prod = PDL_matrix * U_np1over2

    explicit_term = (1 - θ) * ν * PDL_prod + Hδ_hU_np1over2 - F_hM_n
    U_np1 = U_np1over2 - Δt / 2 * explicit_term

    return U_np1
end

# Doesn't seem to work either

#######
function new_HJB_solve(N_h, N_T, M_mat, PDL_matrix, Δt, h, ν, num_iter_HJB, x_grid, δ)
    U_mat = Array{Float64}(undef, N_h, N_T)
    U_mat[:, 1] = Gh_func(M_mat[:, 1])

    #println("Starting HJB_solve ...")
    #HJB_progress = Progress(num_it_MFG, desc="HJB iterations")

    method = ""
    avg_times = [0.0, 0.0, 0.0, 0.0]
    newton_iterations = Float64[]
    hjb_solve_time = time()
    for n in 1:(N_T-1)
        #println("n=",n)
        if method == "explicit"
            U_mat[:, n+1] = explicit_HJB_step(num_iter_HJB, N_h, U_mat[:, n], M_mat[:, n], PDL_matrix, Δt, h, ν, n, x_grid)
        elseif method == "theta"
            U_mat[:, n+1] = imp_exp_HJB_step(num_iter_HJB, N_h, U_mat[:, n], M_mat[:, n], PDL_matrix, Δt, h, ν, n, x_grid)
        else
            U_mat[:, n+1], time_percentages, final_k = new_HJB_step(num_iter_HJB, N_h, U_mat[:, n], M_mat[:, n], PDL_matrix, Δt, h, ν, n, x_grid, δ)
            #println("n: ", n, "Time and time percentages [N (Newtons), J/N, F/N, inverse/N]: ", time_percentages)
            push!(newton_iterations, final_k)
            avg_times[1] += time_percentages[1]
            avg_times[2] += time_percentages[2]
            avg_times[3] += time_percentages[3]
            avg_times[4] += time_percentages[4]
            #println("avg_time: ", avg_times)
        end
        #next!(HJB_progress)
    end

    avg_times[1] = avg_times[1] / (N_T - 1)
    avg_times[2] = avg_times[2] / (N_T - 1)
    avg_times[3] = avg_times[3] / (N_T - 1)
    avg_times[4] = avg_times[4] / (N_T - 1)
    println("HJB solve time: ", time() - hjb_solve_time, ". HJB step avg time and time percentages [N (Newtons), J/N, F/N, inverse/N]: ", avg_times)
    println("Newton iterations, min, max, avg: ", minimum(newton_iterations), ". ", maximum(newton_iterations), ". ", sum(newton_iterations) / length(newton_iterations))
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



function MFG_solve(N_h, N_T, h, Δt, num_it_MFG, num_it_HJB, M_T, α, δ, R, x_grid, write_iteration_results=false)
    M_mat = repeat(Float64.(M_T), 1, N_T)
    #U_mat = Array{Float64}(undef, N_h, N_T)
    U_mat = zeros(Float64, N_h, N_T)
    PDL_matrix = create_PDL_matrix(N_h, α, h, R)

    U_mat_temp = zeros(Float64, N_h, N_T)
    M_mat_temp = zeros(Float64, N_h, N_T)

    println("Starting MFG_solve for loop...")
    ## Didn't work:
    #read_initial_guess = true
    #continue_old_run = true
    for j in 1:num_it_MFG #ProgressBar(1:num_it_MFG)
        mfg_solve_time = time()
        println("j: ", j)
        #println("MFG iterations. j=", j, "/", num_it_MFG)
        #=  if read_initial_guess && j == 1
             #sleep(10)
             run = "notebooks/new_MFG_convergence_rate_runs/run" * string(run_number)
             println("run: ", run)
             println(pwd())
             params = JSON.parsefile(run * "_params.json")
             ind = findfirst(item -> item == h, params["h_list"])
             if ind == 1
                 println("in if ind==1")
                 if continue_old_run
                     println("in if continue_old_run")
                     old_run_num = 22
                     old_run = "notebooks/new_MFG_convergence_rate_runs/run" * string(old_run_num)
                     println("old_run: ", old_run)
                     println(pwd())
                     old_params = JSON.parsefile(old_run * "_params.json")
                     old_h = old_params["h_list"][end]
                     println("old_h: ", old_h)
                     U_old = reverse(readdlm(old_run * "_U_mat_conv_h$(old_h)_deltat$(Δt).csv", ','), dims=2)
                     N_old, N_T2 = size(U_old)
                     #h_ratio = old_h/h # assume h_ratio is power of 2. 
                     @assert N_T == N_T2
                     @assert N_h == 2 * N_old
                     for m in 1:N_T
                         for i in 1:N_old
                             U_mat_temp[2i-1, m] = U_old[i, m]
                         end
                         for i in 1:N_old-1
                             U_mat_temp[2i, m] = (U_old[i, m] + U_old[i+1, m]) / 2
                         end
                         U_mat_temp[end, m] = (U_old[end, m] + U_old[1, m]) / 2
                     end

                 else
                     U_mat_temp = new_HJB_solve(N_h, N_T, M_mat, PDL_matrix, Δt, h, ν, num_it_HJB, x_grid, δ)
                 end
                 println("end if ind==1")
             else
                 println("in else read initial guess")
                 old_h_ind = ind - 1
                 old_h = params["h_list"][old_h_ind]
                 U_old = readdlm(run * "_U_mat_conv_h$(old_h)_deltat$(Δt).csv", ',')
                 N_old, N_T2 = size(U_old)
                 #h_ratio = old_h/h # assume h_ratio is power of 2. 
                 @assert N_T == N_T2
                 @assert N_h == 2 * N_old
                 for m in 1:N_T
                     for i in 1:N_old
                         U_mat_temp[2i-1, m] = U_old[i, m]
                     end
                     for i in 1:N_old-1
                         U_mat_temp[2i, m] = (U_old[i, m] + U_old[i+1, m]) / 2
                     end
                     U_mat_temp[end, m] = (U_old[end, m] + U_old[1, m]) / 2
                 end
                 println("end else read initial guess")
             end
         else =#
        U_mat_temp = new_HJB_solve(N_h, N_T, M_mat, PDL_matrix, Δt, h, ν, num_it_HJB, x_grid, δ)
        #end
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
Δt = 0.001
t_0 = 0
T = 2
ν = 0.09^2
num_it_MFG = 50
num_it_HJB = 20
δ = 0.4
R = 30

###################################+

#### For saving runs ####
runs_folder = "notebooks/new_MFG_convergence_rate_runs/" #Folder to save runs too. NB: Change run number so old runs aren't overwritten.
iterations_folder = "notebooks/new_MFG_iteration_plots/"
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
    #M_T = new_m_T_func(x_grid, 0.5)

    #if write_iters
    push!(params["h_list"], h)
    open(params_file, "w") do io
        JSON.print(io, params)
    end
    #end
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
    #=     if !write_iters
            push!(params["h_list"], h)
            open(params_file, "w") do io
                JSON.print(io, params)
            end
        end =#
end