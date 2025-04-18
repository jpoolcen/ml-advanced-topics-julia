#estimar una proporción con diferentes priors usando el modelo Beta-Binomial, 
#perfecto para entender cómo un prior afecta la inferencia bayesiana, especialmente con pocos datos.
# El prior conjugado de un Binomial es una distribucion beta.
## Prior uniforme se apoya en los datos , el prior es no informativo.
## Curva naranja, el prior ajusta los picos de datos
## Curva verde prior adecuado
## Linea morada el clasico MLE.

using Plots, Distributions, Random, Printf

# Semilla para reproducibilidad
Random.seed!(42)

# Parámetros
θ_real = 0.15
n = 100
datos = rand(Bernoulli(θ_real), n)
x = sum(datos)

# Priors Beta
priors = [
    ("Uniforme Beta(1,1)", Beta(1, 1)),
    ("Optimista Beta(5,2)", Beta(5, 2)),
    ("Conservador Beta(2,8)", Beta(2, 8))
]

# Dominio para θ
θ = 0:0.001:1

# Graficar
plot(title="Posteriors con n = $n y θ_real = $θ_real", xlabel="θ", ylabel="Densidad")

for (label, prior) in priors
    α_post = prior.α + x
    β_post = prior.β + n - x
    posterior = Beta(α_post, β_post)

    # Calcular estadísticas
    media = mean(posterior)
    modo = (α_post - 1) / (α_post + β_post - 2)   # solo válido si α, β > 1
    ci_lower = quantile(posterior, 0.025)
    ci_upper = quantile(posterior, 0.975)

    # Imprimir resumen
    println("===== $label =====")
    @printf "Posterior: Beta(%d, %d)\n" α_post β_post
    @printf "Media posterior: %.4f\n" media
    @printf "MAP (modo): %.4f\n" modo
    @printf "IC 95%%: [%.4f, %.4f]\n" ci_lower ci_upper
    println()

    # Curva posterior
    posterior_vals = pdf.(posterior, θ)
    plot!(θ, posterior_vals, label="$label", lw=2)
end

# Línea de MLE
vline!([x/n], lw=1.5, linestyle=:dash, label="MLE: $(round(x/n, digits=3))")

savefig("posterior_beta_summary_n100.png")
