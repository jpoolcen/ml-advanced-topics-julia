# 📘 Probabilistic ML - Murphy (Notas y Ejercicios en Julia)

Este repositorio contiene mis notas personales, visualizaciones y ejercicios resueltos en **Julia** basados en el libro  
**"Probabilistic Machine Learning: Advanced Topics"** de Kevin P. Murphy (2023).

La intención es comprender a profundidad los conceptos de inferencia bayesiana, modelos generativos y aprendizaje profundo probabilístico, implementando ejercicios desde cero y documentando el proceso.

---

## 📂 Estructura del Repositorio

En el repositorio puede encontrar seccion acordes a los topicos cubiertos.
- 📂 **bayesian**: este directorio contiene ejercicios de estimación de Bayes,distribución a posteriori, MLE (Maximum Likelihood Estimation) y MAP (Maximun a Posteriori). En cada ejercicio se anexa una visualizacion. En los ejercicios se consideran datos sintéticos sobre salud mental. Contiene un archivo **bayesian.jl** y las visualizaciones se colocan en un directorio denominado output.

Recordando el **Teorema de Bayes**

P(H|D)=P(D|H)P(H)/P(D)

H: Hipotesis
D: Datos.

En el contexto de salud mental puede ser:
Probabilidad de que un paciente tenga una enfermedad dado un resultado positivo en la prueba
 P(E|T) = P(T|E) * P(E) / P(T)
 . 
P(E|T): Posterior, lo que queremos calcular

P(T|E): Likelihood, Verosimilitud, observar los datos dados los parametros o hipotesis 

P(E): Prior (creencia), P(D): Evidence

P(D): Evidence: probabilidad total de observar los datos (normalizante)

Algunas conclusiones que obtuve en esta sección son:
1. MLE estima el valor del parámetro que maximiza la verosimilitud. Ignora la "creencia", esto es no incorpora información a priori sobre los parámetros.
2. MAP estima el valor que maximiza la distribución a posteriori. Incorpora la información a priori. Si no hay priori o es uniforme, el MAP equivale a MLE. En MAP si hay poca información o pocos datos, el prior tiene mayor peso. Si hay mucha información la posterior se parece a MLE.

-📂 **Divergence** En este directorio se presentan los conceptos de divergencia entre distribuciones de probabilidad, partiendo de la definicion general de f-divergence y generando las medidas de divergencias para comparar distribuciones de probabilidad.

Este tema es relevante para **inferencia variacional**, modelos generativos y aprendizaje profundo probabilistico.
Para comparar distribuciones, se basa en la idea del ratio r(x) = p(x)/q(x). Para mayor profundidad puede consultar la seccion 2.7.1 de la referencia de Murphy.

Algunas medidas de divergecia utilizadas son:
1. Divergencia de Kullback-Leibler. (Inferencia Variacional) 
2. Divergencia de Jensen Shannon
3. Total Variation (TV) Distance
4. Hellinger Distance.


Las medidas de divergencias son utiles para: evaluar qué tan cerca estás de la distribución verdadera, medir la pérdida de información al usar una aproximación,hacer optimización sobre distribuciones (como en VAEs, GANs, Bayesian inference).

Nota: Si dos distribuciones son iguales, esto es P(x)/Q(x)=1, entonces las medidas de divergencia es 0.

Se muestran 3 ejemplos donde se simulan las aproximaciones y la estimacion de las divergencias. Además se incluye una simulación sobre la estimación de la distribución de la prevalencia en municipios con datos ficticios y posteriormente se obtienen las metricas de divergencia.

- 📂 **Conjugate Priors**: este directorio contiene ejercicios sobre los prior.
Un prior representa tu creencia sobre los parámetros antes de observar los datos. Existen tipos de prior: Informativo (conocimiento previo), No informativo, conjugado, jerarquico y funcional.

Un prior conjugado respecto a una verosimilitud es aquel tal que la posterior es de la misma familia que el prior.

Esto simplifica los cálculos y permite obtener soluciones cerradas.

Un prior empirico se puede estimar con Jefrey y un funcional que utilice Fisher Information.

Un prior informativo se puede tomar a partir de estudios previos.

Jerarquicos en el cual los parametros del prior son aleatorios.

## Optimizacion Bayesiana

En el libro de Murphy podemos encontrar las fundamentos matematicos de esta tecnica. 
En ejemplo propuesto consiste en maximizar la tasa de detección de casos de salud mental
Supón que tienes un sistema que detecta casos según un umbral de puntuación 
x (entre 0 y 1), y quieres encontrar el mejor umbral para maximizar detecciones reales.

No conoces cómo se comporta la tasa real f(x), pero puedes evaluarla costosa o parcialmente (por simulación, estudio piloto, etc.).
En Julia se utiliza Surrogates.jl para procesos gaussianos y BayesianOptimization.jl para el ciclo de optimizacion.
