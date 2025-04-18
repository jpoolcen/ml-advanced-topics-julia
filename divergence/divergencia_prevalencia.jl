using Plots, Distributions, StatsBase

# Simular prevalencias reales (distribución P)
n_municipios = 20
p_real = rand(Beta(2, 5), n_municipios)  # Distribución más sesgada hacia valores bajos

# Simular predicción modelo (distribución Q con error o sesgo)
p_predicho = p_real .+ rand(Normal(0, 0.05), n_municipios)
p_predicho = clamp.(p_predicho, 0.0001, 0.9999)  # Evitar valores extremos

# Normalizar como si fueran distribuciones discretas (suman 1)
P = p_real ./ sum(p_real)
Q = p_predicho ./ sum(p_predicho)

# Calcular divergencias

# 1. KL(P‖Q)
kl_div = sum(P .* log.(P ./ Q))

# 2. JS
M = 0.5 .* (P + Q)
js_div = 0.5 * sum(P .* log.(P ./ M)) + 0.5 * sum(Q .* log.(Q ./ M))

# 3. Hellinger
hellinger = sqrt(0.5 * sum((sqrt.(P) .- sqrt.(Q)).^2))

# 4. Total Variation
tv = 0.5 * sum(abs.(P .- Q))

# Mostrar resultados
println("=== DIVERGENCIAS ENTRE PREVALENCIAS (REALS vs. PREDICHAS) ===")
println("KL(P || Q)      = ", round(kl_div, digits=4))
println("JS Divergence   = ", round(js_div, digits=4))
println("Hellinger Dist  = ", round(hellinger, digits=4))
println("Total Variation = ", round(tv, digits=4))

# Visualización
bar(1:n_municipios, [P Q], label=["Real" "Modelo"], legend=:topleft)
xlabel!("Municipios")
ylabel!("Prevalencia (normalizada)")
title!("Comparación de prevalencia real vs. modelo")
savefig("divergencia_prevalencia.png")
