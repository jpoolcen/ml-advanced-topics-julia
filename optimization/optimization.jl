using BayesianOptimization
using GaussianProcesses
using Random
using Plots

# Semilla para reproducibilidad
Random.seed!(1234)

# Función objetivo simulada (tasa de detección)
f(x) = exp(-50 * (x[1] - 0.65)^2) + 0.05 * randn()

# Configurar el modelo de GP
meanfunc = MeanConst(0.0)
kernel = SEArd([0.0], 0.0)
logNoise = log(0.05)
model = ElasticGPE(1, mean=meanfunc, kernel=kernel, logNoise=logNoise)

# Optimización de hiperparámetros del modelo
modeloptimizer = MAPGPOptimizer(every=50,
                                noisebounds=[-4.0, 3.0],
                                kernbounds=[[-1.0, 0.0], [4.0, 10.0]],
                                maxeval=40)

# Inicializar la optimización bayesiana
opt = BOpt(f,
           model,
           ExpectedImprovement(),
           modeloptimizer,
           [0.0], [1.0],
           repetitions=1,
           maxiterations=20,
           sense=Max,
           verbosity=Progress)

# Ejecutar optimización
result = boptimize!(opt)

# Mostrar resultados
println("\n== Resultado final ==")
println("Mejor x estimado: ", result.model_optimizer[1])
println("Mejor valor estimado: ", result.model_optimum)

# Visualizar puntos evaluados usando opt.trace
#xs = [trace.x[1] for trace in opt.trace ]
#ys = [trace.value for trace in opt.trace]

#scatter(xs, ys, xlabel="Umbral", ylabel="Tasa de detección", label="Evaluaciones",
#title="Optimización Bayesiana con BayesianOptimization.jl")
#savefig("bayesopt_deteccion.png")
