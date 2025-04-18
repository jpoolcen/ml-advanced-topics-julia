using Plots, Distributions, Printf

# 1. Simular datos de encuesta
n = 10                     # tamaño de muestra (pequeño para ver efecto del prior)
θ_real = 0.2               # prevalencia real
x = rand(Binomial(n, θ_real))  # simulación de respuestas positivas

println("Simulación: x = $x positivos de n = $n (θ_real = $θ_real)")

# 2. Definir los priors
priors = [
    ("Uniforme Beta(1,1)", Beta(1 + x, 1 + n - x)),
    ("Jeffreys Beta(0.5,0.5)", Beta(0.5 + x, 0.5 + n - x))
]

# 3. Dominio para θ
θ = 0:0.001:1

# 4. Graficar posteriors
plot(title="Comparación posterior: Uniforme vs. Jeffreys (n=$n, x=$x)",
     xlabel="θ (prevalencia)", ylabel="Densidad")

for (label, posterior) in priors
    plot!(θ, pdf.(posterior, θ), label=label, lw=2)

    # Calcular y mostrar media, MAP, intervalo creíble
    media = mean(posterior)
    modo = (posterior.α - 1) / (posterior.α + posterior.β - 2)
    ci_lower = quantile(posterior, 0.025)
    ci_upper = quantile(posterior, 0.975)

    println("\n=== $label ===")
    @printf "Media posterior: %.4f\n" media
    @printf "MAP (modo): %.4f\n" modo
    @printf "IC 95%%: [%.4f, %.4f]\n" ci_lower ci_upper
end

vline!([x/n], lw=1.5, linestyle=:dash, label="MLE: $(round(x/n, digits=3))")

savefig("posterior_jeffreys_vs_uniforme.png")
