using Plots, Distributions, StatsBase

# Crear dos distribuciones normales, varianza igual y con media diferente
μ₁, σ₁ = 0.0, 1.0
μ₂, σ₂ = 1.0, 1.0

p = Normal(μ₁, σ₁)
q = Normal(μ₂, σ₂)

# Dominio de evaluación
x = -5:0.01:5
p_vals = pdf.(p, x)
q_vals = pdf.(q, x)

# Normalizar para asegurar que suman 1 en caso de discretización
p_vals ./= sum(p_vals)
q_vals ./= sum(q_vals)

# 1. KL Divergence D_KL(P || Q)
kl_div = sum(p_vals .* log.(p_vals ./ q_vals))

# 2. JS Divergence
m_vals = 0.5 .* (p_vals + q_vals)
js_div = 0.5 * sum(p_vals .* log.(p_vals ./ m_vals)) +
         0.5 * sum(q_vals .* log.(q_vals ./ m_vals))

# 3. Hellinger distance
hellinger = sqrt(0.5 * sum((sqrt.(p_vals) .- sqrt.(q_vals)).^2))

# 4. Total Variation
tv = 0.5 * sum(abs.(p_vals .- q_vals))

# Imprimir resultados
println("===== DIVERGENCIAS ENTRE P Y Q =====")
println("KL(P || Q)        = ", round(kl_div, digits=4))
println("Jensen-Shannon    = ", round(js_div, digits=4))
println("Hellinger         = ", round(hellinger, digits=4))
println("Total Variation   = ", round(tv, digits=4))

# Visualización
plot(x, p_vals, label="P ~ N(0,1)", lw=2)
plot!(x, q_vals, label="Q ~ N(1,1)", lw=2, linestyle=:dash)

# Rellenar la diferencia |p - q| para visualizar TV
diff_vals = abs.(p_vals .- q_vals)
plot!(x, diff_vals, fillrange=0, fillalpha=0.2, color=:gray, label="|P - Q| (TV área)")

xlabel!("x")
ylabel!("Densidad")
title!("Comparación de distribuciones y divergencias")
savefig("divergencias_gaussianas.png")