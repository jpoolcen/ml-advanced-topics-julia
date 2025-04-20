### Módulo 1: Generación de datos sintéticos
module Cavi

using Distributions, Random, LinearAlgebra, Plots
export generate_data, initialize_variational, cavi_iteration, show_cluster_assignments, compute_elbo, plot_elbo


function generate_data(N::Int, K::Int, mu_true::Vector{Float64})
    sigma = 1.0
    z = rand(1:K, N)  # asignaciones verdaderas
    x = [rand(Normal(mu_true[zi], sigma)) for zi in z]
    return x, z
end

### Módulo 2: Inicialización de parámetros variacionales
function initialize_variational(N::Int, K::Int)
    r = fill(1.0/K, N, K)          # responsabilidades (q(z_n))
    m = randn(K)                   # medias de q(mu_k)
    s2 = ones(K)                   # varianzas de q(mu_k)
    return r, m, s2

end

### Módulo 3: Ciclo de actualizaciones CAVI
function cavi_iteration(x, r, m, s2; τ=1.0, max_iter=100)
    
    N, K = size(r)
    elbos = Float64[]
    for iter in 1:max_iter
        # 1. Actualizar responsabilidades r_{nk}
        for n in 1:N
            for k in 1:K
                log_rnk = -0.5 * ((x[n]^2 - 2*x[n]*m[k] + m[k]^2 + s2[k]))
                r[n, k] = log_rnk
            end
            # Normalizar
            r[n, :] .= exp.(r[n, :] .- maximum(r[n, :]))  # estabilidad
            r[n, :] ./= sum(r[n, :])
        end

        # 2. Actualizar parámetros de q(mu_k)
        for k in 1:K
            N_k = sum(r[:, k])
            s2[k] = 1.0 / (τ + N_k)
            m[k] = s2[k] * sum(r[:, k] .* x)
        end
        push!(elbos, compute_elbo(x, r, m, s2; τ=τ))
    end
    return r, m, s2,elbos
end


### Módulo 4: Visualización de resultados
function show_cluster_assignments(r, x, z_true; save_path="")
    z_hat = map(i -> argmax(r[i, :]), 1:size(r, 1))
    plt = scatter(x, z_hat, label="CAVI clusters", title="Asignaciones de Clusters por CAVI", xlabel="x", ylabel="Cluster", legend=:topright)
    scatter!(plt, x, z_true, label="Clusters verdaderos", markershape=:x)
    if save_path != ""
        savefig(plt, save_path)
        println("Gráfico guardado en: ", save_path)
    end
    display(plt)
end


function compute_elbo(x, r, m, s2; τ=1.0)
    N, K = size(r)
    elbo = 0.0
    for n in 1:N
        for k in 1:K
            log_p_x = -0.5 * ((x[n]^2 - 2*x[n]*m[k] + m[k]^2 + s2[k]))
            log_q_z = log(r[n, k] + 1e-12)  # evitar log(0)
            elbo += r[n, k] * (log_p_x - log_q_z)
        end
    end
    for k in 1:K
        elbo += -0.5 * (τ * (m[k]^2 + s2[k]) - log(s2[k]) - 1)
    end
    return elbo
end

function plot_elbo(elbos; save_path="")
    plt = plot(elbos, xlabel="Iteración", ylabel="ELBO", title="Evolución de la ELBO", legend=false)
    if save_path != ""
        savefig(plt, save_path)
        println("Gráfico de ELBO guardado en: ", save_path)
    end
    display(plt)
end


end
