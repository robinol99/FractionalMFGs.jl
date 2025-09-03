using ProgressBars;
using Infiltrator


using LinearAlgebra, Plots, FFTW;
function plus(x)
    return max(x, 0)
end

function minus(x)
    return max(-x, 0)
end

function create_D_vecs(u_vec, t_n, h, C_H) # returns Dplusminus and Dminusplus vectors
    Dplus = [diff(u_vec); 0] ./ h
    Dminus = [0; diff(u_vec)] ./ h
    Dplus[end] = (u_vec[1] - u_vec[end]) / h
    Dminus[1] = (u_vec[1] - u_vec[end]) / h
    Dplusminus = minus.(Dplus)
    Dminusplus = plus.(Dminus)
    Dplusplus = plus.(Dplus)
    Dminusminus = minus.(Dminus)
    return (Dplusminus, Dminusplus, Dplusplus, Dminusminus)
end

function g_func(u_vec, t_n, h, C_H)
    (Dplusminus, Dminusplus, temp, temp) = create_D_vecs(u_vec, t_n, h, C_H)
    return C_H .* (Dplusminus .^ 2 + Dminusplus .^ 2)
end

function create_Dg_mat(v, t_n, N_h, h, C_H) # verified is correct
    (Dplusminus, Dminusplus, temp, temp) = create_D_vecs(v, t_n, h, C_H)
    Dg_mat = zeros(N_h, N_h)
    tridiag = Tridiagonal(-Dminusplus[2:end], Dplusminus + Dminusplus, -Dplusminus[1:end-1])
    Dg_mat[1, N_h] = -Dminusplus[1]
    Dg_mat[N_h, 1] = -Dplusminus[end]
    Dg_mat += tridiag
    Dg_mat *= 2 / h
    return C_H .* Dg_mat
end
using SpecialFunctions: gamma, loggamma, zeta
function TS(m, α, N_h, R)
    if m == 0
        return 0
    end
    K_α = 0.0
    for ν in -R:1:R
        K_α += exp(loggamma(abs(m - N_h * ν) - α / 2) - loggamma((abs(m - N_h * ν) + 1 + α / 2)))
    end
    return K_α
end


#function create_DPL_matrix(N_h, α, h, R)
#    DPL_mat = Matrix{Float64}(undef, N_h, N_h)
#    zeta_term= zeta(1+α)
#    for i in 1:N_h
#        for j in 1:N_h
#            DPL_mat[i,j] = TS(abs(i-j), α, N_h, R)
#        end
#    end
#    for γ in 1:N_h
#        DPL_mat[γ, γ] = -(sum([TS(β-γ, α, N_h, R) for β in 1:N_h]) + 2*zeta_term - 
#        sum([1/k^(1+α) for k in 1:((R+1)*N_h - γ) ]) - sum([1/k^(1+α) for k in 1:(R*N_h + γ - 1) ]))
#    end
#    c_α = 2^α * gamma((1+α)/2) / (√π * abs(gamma(-α/2)) )
#    return -c_α / (h^α) * DPL_mat
#end



function create_DPL_matrix(N_h, α, h, R) # new method, spectral method. Verified agains old method, and gives zero row sums.
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


function tranport_matrix(U, t_n, N_h, h, C_H) # verified!
    (DPM, DMP, DPP, DMM) = create_D_vecs(U, t_n, h, C_H)
    TU_mat = zeros(N_h, N_h)
    tridiag = Tridiagonal(DMM[2:end], -DPM - DMP, DPP[1:end-1])
    TU_mat[1, N_h] = DMM[1]
    TU_mat[N_h, 1] = DPP[end]
    # tridiag = Tridiagonal(DMM[1:end-1], -(DMP + DPM), DPP[2:end])
    # TU_mat[1, N_h] = DMM[end]
    # TU_mat[N_h, 1] = DPP[1]
    TU_mat += tridiag
    TU_mat *= 2 / h # not by h
    return C_H .* TU_mat
end


function mathcalF(U_np1, U_n, M_n, DPL_matrix, x_vec, n, N_h, Δt, h, ν, α, R, δ, C_H)
    g_part = g_func(U_np1, n * Δt, h, C_H)
    F_part = -F_h(M_n, x_vec, 2 - n * Δt, δ)
    return U_np1 - U_n + Δt * (ν * DPL_matrix * U_np1 + +g_part + F_part)
end

function J_F(x, DPL_matrix, t_n, N_h, Δt, h, ν, α, R, C_H) # verified is correct
    Dg_mat = create_Dg_mat(x, t_n, N_h, h, C_H)
    return 1.0I(N_h) + Δt * (ν * DPL_matrix + Dg_mat)
end

function HJB_step(U_n, M_n, DPL_matrix, num_it_HJB, x_vec, n, N_h, Δt, h, ν, α, R, δ, C_H)
    U_np1 = U_n
    #=    if n > 69
           println("Inside")
           @infiltrate
       end
    =#
    for _ in num_it_HJB
        jacobi = J_F(U_np1, DPL_matrix, n * Δt, N_h, Δt, h, ν, α, R, C_H)
        F_vec = mathcalF(U_np1, U_n, M_n, DPL_matrix, x_vec, n, N_h, Δt, h, ν, α, R, δ, C_H)
        δ = jacobi \ F_vec
        U_np1 = U_np1 - δ
        #=         if n > 69
                    @infiltrate
                end =#
    end
    return U_np1
end

function HJB_solve(M_mat, num_it_HJB, x_vec, N_h, Δt, N_T, h, ν, α, R, δ, C_H)
    U_mat = Array{Float64}(undef, N_h, N_T)
    U_mat[:, 1] = G_h(M_mat[:, 1])
    DPL_matrix = create_DPL_matrix(N_h, α, h, R)
    #println("DPL_matrix: ", DPL_matrix)
    for n in 1:(N_T-1)
        U_mat[:, n+1] = HJB_step(U_mat[:, n], M_mat[:, n], DPL_matrix, num_it_HJB, x_vec, n, N_h, Δt, h, ν, α, R, δ, C_H)
    end
    return U_mat
end

function FPK_step(U_np1, M_np1, n, DPL_mat, N_h, Δt, h, ν, α, R, C_H) # not exactly verified, but DPL and TU is verified.
    TU = tranport_matrix(U_np1, n * Δt, N_h, h, C_H)
    total_mat = 1.0I(N_h) + Δt * (ν * DPL_mat - TU)
    M_n = total_mat \ M_np1
    @infiltrate
    return M_n
end
function FPK_solve(U_mat, M_T, N_h, Δt, N_T, h, ν, α, R, C_H)
    M_mat = Array{Float64}(undef, N_h, N_T)
    M_mat[:, end] = M_T
    DPL_mat = create_DPL_matrix(N_h, α, h, R)
    for j in 1:(N_T-1)
        M_mat[:, N_T-j] = FPK_step(U_mat[:, N_T-j+1], M_mat[:, N_T-j+1], N_T - j + 1, DPL_mat, N_h, Δt, h, ν, α, R, C_H)
    end
    return M_mat
end

function MFG_solve(M_T, cv)
    (α, h, N_h, Δt, N_T, ν, num_it_MFG, num_it_HJB, x_vec, R, δ, C_H) = cv
    M_mat = Array{Float64}(undef, N_h, N_T)
    M_mat .= M_T
    U_mat = Array{Float64}(undef, N_h, N_T)
    for _ in ProgressBar(1:num_it_MFG)
        U_mat = HJB_solve(M_mat, num_it_HJB, x_vec, N_h, Δt, N_T, h, ν, α, R, δ, C_H)
        M_mat = FPK_solve(U_mat, M_T, N_h, Δt, N_T, h, ν, α, R, C_H)
        @infiltrate
    end
    return (U_mat, M_mat)
end

using QuadGK;

function ϕ_δ(x, δ)
    1 / (δ * sqrt(2 * π)) * exp(-x^2 / (2 * δ^2))
end

# function conv_term(m_vec, x_vec, δ)
#     h = x_vec[2] - x_vec[1]
#     N_h = length(m_vec)
#     conv_vec = Vector{Float64}(undef, N_h)
#     for j in 1:N_h
#         conv_vec[j] = h*sum( [ m_vec[i] * ϕ_δ(x_vec[j] - x_vec[i],δ) for i in 1:N_h]) # * h
#     end
#     return conv_vec
# end

function conv_term(m_vec, x_vec, δ)
    N_h = length(m_vec)
    h = x_vec[2] - x_vec[1]
    conv_vec = Vector{Float64}(undef, N_h)

    # Base offsets from reference node and minimum-image wrap
    # Δs = x_vec .- x_vec[1]
    # Δs .-= round.(Δs)  # map to (-0.5, 0.5]
    Δs = (x_vec .- x_vec[1]) .- round.(x_vec .- x_vec[1])  # (-0.5, 0.5]

    # Periodic kernel weights and discrete normalization
    weights = ϕ_δ.(Δs, δ)
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

# f_func = (x, t) -> cos(2*pi*(x-t^2))^2
f_func = (x, t) -> 5 * (x − 0.5(1 − sin(2 * π * t)))^2

function F_h(M, x_vec, t_n, δ)
    ϕm = conv_term(M, x_vec, δ)
    fⁿ = [f_func(x_i, t_n) for x_i in x_vec]
    return ϕm + fⁿ
end

### INITIAL AND TERMINAL CONDITIONS

function G_h(M)
    return zeros(length(M))
end

# m_T_func_unnorm = (x) -> exp(-50*(x-0.5)^2)
m_T_func_unnorm = (x) -> exp(-(x - 0.5)^2 / 0.1^2)
oneOverC, error = quadgk(m_T_func_unnorm, 0, 1);

function m_T_func(x)
    return 1 / oneOverC * m_T_func_unnorm(x)
end




#RUN SIMULATION

using DelimitedFiles;
println("START")
### Solution for different h

##########
run = "MFG_convergence_rate_runs/run9" #Folder to save runs too. NB: Change run number so old runs aren't overwritten.
##########

####Choose grid sizes:
h_reference = 1 / 2^9
h_list = [1 / 2^7, 1 / 2^8, h_reference]
###################################+

M_list = Array{Float64}[]
U_list = Array{Float64}[]

println("hlist: ", h_list)
println("h_reference: ", h_reference)
println("M_list: ", M_list)
println("U_list", U_list)

for h in h_list
    println("")
    println("start for loop, h=", h)
    ####### NB Might take some time to run

    ####Select the right parameters: 
    #NB: Add the parameters for each new run to the MFG_convergence_runs_parameters.ipynb file so we know what 
    #parameters were used for each run.
    α = 1.5
    x_vec = -1:h:(2-h)
    N_h = length(x_vec)
    Δt = 0.001
    t_vec = 0:Δt:(1-Δt)
    N_T = length(t_vec)
    ν = 0.09^2
    num_it_MFG = 50
    num_it_HJB = 20
    δ = 0.4
    R = 30
    C_H = 0.5
    cv = (α, h, N_h, Δt, N_T, ν, num_it_MFG, num_it_HJB, x_vec, R, δ, C_H) # create a constant-vector, to avoid clutter for all constants we need.
    println("C_H: ", C_H)
    #terminal condition
    M_T = m_T_func.(x_vec)
    ################################

    println("Running MFG_solve with h=", h)
    (U_mat, M_mat) = MFG_solve(M_T, cv)
    M_mat = reverse(M_mat, dims=2)
    U_mat = reverse(U_mat, dims=2)
    ############

    println("Done running MFG_sovle with h=", h)
    #println("M_mat: ")
    #println(M_mat)

    println("size(M_mat): ", size(M_mat))
    println("size(U_mat): ", size(U_mat))

    push!(M_list, M_mat)
    push!(U_list, U_mat)
    println("!!!!!!!!!!!!!!")
    writedlm(run * "_U_mat_convergence_h_" * string(h) * "deltat_" * string(Δt) * ".csv", U_mat, ",")
    writedlm(run * "_M_mat_convergence_h_" * string(h) * "deltat_" * string(Δt) * ".csv", M_mat, ",")
end

println("DONE")
println("size(M_list): ", size(M_list))
println("size(U_list): ", size(U_list))