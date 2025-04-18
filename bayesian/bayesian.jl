# Teorema de bayes
# Probabilidad de que un paciente tenga una enfermedad dado un resultado positivo en la prueba
# P(E|T) = P(T|E) * P(E) / P(T)
# P(H|D) = (P(D|H) * P(H)) / P(D). 
# P(H|D): Posterior, lo que queremos calcular
# P(D|H): Likelihood, Verosimilitud, observar los datos dados los parametros o hipotesis 
# P(H): Prior (creencia), P(D): Evidence
## Notes by jpool. Bibliography: Murphy, K. P. (2012). Machine Learning: A Probabilistic Perspective. MIT Press.
using Plots,Distributions

function bayes_example()
    prior = 0.01 # P(H) o P(Enfermo), enfermo
    sensitivity = 0.95  # P(Positivo|Enfermo) - Probabilidad de que la prueba sea positiva dado que el paciente tiene la enfermedad
    specificity = 0.90  # P(~D|~H o P(~T|~E)), P(Negativo | Sano) - Probabilidad de que la prueba sea negativa dado que el paciente no tiene la enfermedad

    #P(Positivo) = P(Positivo|Enfermo) * P(Enfermo) + P(Positivo|No Enfermo) * P(No Enfermo)

    p_positive = sensitivity * prior + (1 - specificity) * (1 - prior) # P(T) o P(D)

    # Posterior
    posterior = (sensitivity * prior) / p_positive # P(E|T) o P(H|D)
    return posterior
end

function posterior_probability(prior::Float64, sensitivity::Float64, specificity::Float64)
    # Definir los parámetros
    p_positive = sensitivity * prior + (1 - specificity) * (1 - prior) # P(T) o P(D)
    # Posterior
    posterior = (sensitivity * prior) / p_positive # P(E|T) o P(H|D)
    return posterior
end

# Ejemplo de uso
post_probability = bayes_example()
println("La probabilidad de que el paciente tenga la enfermedad dado un resultado positivo en la prueba es: ", post_probability)

# ejemplo de simulación de bayes variando la prevalencia.
## La prevalencia es la "carga" de la enfermedad en la población, es decir, 
## la proporción de personas que tienen la enfermedad en un momento dado.
sensitivity = 0.99
specificity = 0.95

# rango de prevalencia
priors = range(0.01, 0.5, length=200) # P(H)
# rango de probabilidades
posteriors = [posterior_probability(p, sensitivity, specificity) for p in priors] # P(E|T)

# graficar 
plot(priors .*100, posteriors.*100, xlabel="Prevalencia (%)",
  ylabel="Probabilidad Posterior (P(E|T) % )", title="Valor predictivo positivo vs Prevalencia", legend=false,lw=2,grid=true)
savefig("output/posterior_vs_prevalencia.png")




### MLE vs MAP ----------

# MLE: Maximum Likelihood Estimation
# MAP: Maximum A Posteriori Estimation
# MLE: P(D|H) o P(T|E)
# MAP: P(H|D) o P(E|T)
# Datos observados
n = 10                # Total de personas
x = 8                 # Personas con ansiedad

# Espacio de parámetros p (de 0 a 1)
p_range = 0:0.001:1

# Likelihood (frecuentista)
likelihood = [pdf(Binomial(n, p), x) for p in p_range]

# Prior bayesiano Beta(2, 2)
prior = [pdf(Beta(2, 2), p) for p in p_range]

# Posterior ~ Beta(alpha + x, beta + n - x)
posterior = [pdf(Beta(2 + x, 2 + n - x), p) for p in p_range]

# Normalizar todas para comparación
likelihood ./= maximum(likelihood)
prior ./= maximum(prior)
posterior ./= maximum(posterior)


plot(p_range, likelihood, label="Verosimilitud (MLE)", lw=2)
plot!(p_range, prior, label="Prior Beta(2,2)", lw=2, ls=:dash)
plot!(p_range, posterior, label="Posterior (MAP)", lw=2, color=:black)
xlabel!("Proporción p")
ylabel!("Densidad (normalizada)")
title!("Comparación MLE vs MAP")
savefig("output/mle_vs_map.png")
### Si hay pocos datos el prior pesa mas en el posterior, si hay mucha informacion el posterior se parece al likelihood.
### El MLE no utiliza informacion a prior

# MLE vs MAP extendido

# Datos observados
n = 10                # Total de personas
x = 8                 # Personas con ansiedad

# Prior bayesiano Beta(α, β)
α = 2
β = 2

# Posterior: Beta(α + x, β + n - x)
posterior_α = α + x
posterior_β = β + n - x
posterior_dist = Beta(posterior_α, posterior_β)

# MLE: proporción directa
p_mle = x / n

# MAP: modo de la Beta posterior
p_map = (posterior_α - 1) / (posterior_α + posterior_β - 2)

# Media de la posterior
p_mean = mean(posterior_dist)

println("====== ESTIMACIONES ======")
println("MLE (frecuentista):        p̂ = ", round(p_mle, digits=3))
println("MAP (modo posterior):      p̂ = ", round(p_map, digits=3))
println("Media posterior (esperada):p̂ = ", round(p_mean, digits=3))
println("Posterior ~ Beta($posterior_α, $posterior_β)")
println("===========================")

# Graficar verosimilitud, prior y posterior
p_range = 0:0.001:1
likelihood = [pdf(Binomial(n, p), x) for p in p_range]
prior = [pdf(Beta(α, β), p) for p in p_range]
posterior = [pdf(posterior_dist, p) for p in p_range]

likelihood ./= maximum(likelihood)
prior ./= maximum(prior)
posterior ./= maximum(posterior)

plot(p_range, likelihood, label="Verosimilitud (MLE)", lw=2)
plot!(p_range, prior, label="Prior Beta(2,2)", lw=2, ls=:dash)
plot!(p_range, posterior, label="Posterior (MAP)", lw=2, color=:black)
xlabel!("Proporción p")
ylabel!("Densidad (normalizada)")
title!("Comparación MLE vs MAP")

savefig("output/mle_vs_map_extendido.png")


### Ejemplo de MLE y MAP con Intervalos de confiabilidad

# Datos observados
n = 10
x = 8

# Prior Beta(α, β)
α = 2
β = 2

# Posterior: Beta(α + x, β + n - x)
posterior_α = α + x
posterior_β = β + n - x
posterior_dist = Beta(posterior_α, posterior_β)

# Estimaciones puntuales
p_mle = x / n
p_map = (posterior_α - 1) / (posterior_α + posterior_β - 2)
p_mean = mean(posterior_dist)

# Intervalo de credibilidad del 95%
ci_lower = quantile(posterior_dist, 0.025)
ci_upper = quantile(posterior_dist, 0.975)

println("====== ESTIMACIONES ======")
println("MLE (frecuentista):         p̂ = ", round(p_mle, digits=3))
println("MAP (modo posterior):       p̂ = ", round(p_map, digits=3))
println("Media posterior esperada:   p̂ = ", round(p_mean, digits=3))
println("Posterior ~ Beta($posterior_α, $posterior_β)")
println("Intervalo de credibilidad 95%: [", round(ci_lower, digits=3), ", ", round(ci_upper, digits=3), "]")
println("===========================")

# Rango para el plot
p_range = 0:0.001:1
likelihood = [pdf(Binomial(n, p), x) for p in p_range]
prior = [pdf(Beta(α, β), p) for p in p_range]
posterior = [pdf(posterior_dist, p) for p in p_range]

# Normalización para comparación visual
likelihood ./= maximum(likelihood)
prior ./= maximum(prior)
posterior ./= maximum(posterior)

# Curvas
plot(p_range, likelihood, label="Verosimilitud (MLE)", lw=2)
plot!(p_range, prior, label="Prior Beta(2,2)", lw=2, ls=:dash)
plot!(p_range, posterior, label="Posterior (MAP)", lw=2, color=:black)

# Intervalo de credibilidad (franja gris)
vspan!(Tuple([ci_lower, ci_upper]); color=:gray, alpha=0.2)

# Ejes
xlabel!("Proporción p")
ylabel!("Densidad (normalizada)")
title!("Comparación MLE vs MAP")


savefig("output/mle_vs_map_credibilidad.png")