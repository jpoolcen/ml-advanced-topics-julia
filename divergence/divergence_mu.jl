using Plots, Distributions

# Distribución base P ~ N(0, 1)
p = Normal(0.0, 1.0)
x = collect(-10:0.01:10)

# Función para calcular divergencias
function divergencias(p::Distribution, q::Distribution, x::Vector)
    p_vals = pdf.(p, x)
    q_vals = pdf.(q, x)

    # Normalizar (como si fueran distribuciones discretas)
    p_vals ./= sum(p_vals)
    q_vals ./= sum(q_vals)

    # KL(P || Q)
    kl = sum(p_vals .* log.(p_vals ./ q_vals))

    # JS divergence
    m = 0.5 .* (p_vals + q_vals)
    js = 0.5 * sum(p_vals .* log.(p_vals ./ m)) + 0.5 * sum(q_vals .* log.(q_vals ./ m))

    # Hellinger
    hellinger = sqrt(0.5 * sum((sqrt.(p_vals) .- sqrt.(q_vals)).^2))

    # Total Variation
    tv = 0.5 * sum(abs.(p_vals .- q_vals))

    return kl, js, hellinger, tv
end

# Rango de medias de Q
mu_q_vals = -3.0:0.1:3.0
kl_vals, js_vals, hell_vals, tv_vals = [], [], [], []

for μq in mu_q_vals
    q = Normal(μq, 1.0)
    kl, js, h, tv = divergencias(p, q, x)
    push!(kl_vals, kl)
    push!(js_vals, js)
    push!(hell_vals, h)
    push!(tv_vals, tv)
end


# Graficar
plot(mu_q_vals, kl_vals, label="KL(P || Q)", lw=2)
plot!(mu_q_vals, js_vals, label="JS", lw=2)
plot!(mu_q_vals, hell_vals, label="Hellinger", lw=2)
plot!(mu_q_vals, tv_vals, label="Total Variation", lw=2)
xlabel!("Media de Q (μ_q)")
ylabel!("Divergencia")
title!("Divergencias vs desplazamiento de la media")
savefig("divergencias_vs_mu.png")
