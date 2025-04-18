# 📘 Probabilistic ML - Murphy (Notas y Ejercicios en Julia)

Este repositorio contiene mis notas personales, visualizaciones y ejercicios resueltos en **Julia** basados en el libro  
**"Probabilistic Machine Learning: Advanced Topics"** de Kevin P. Murphy (2023).

La intención es comprender a profundidad los conceptos de inferencia bayesiana, modelos generativos y aprendizaje profundo probabilístico, implementando ejercicios desde cero y documentando el proceso.

---

## 📂 Estructura del Repositorio

En el repositorio puede encontrar seccion acordes a los topicos cubiertos.
- **bayesian**: este directorio contiene ejercicios de estimación de Bayes,distribución a posteriori, MLE (Maximum Likelihood Estimation) y MAP (Maximun a Posteriori). En cada ejercicio se anexa una visualizacion. En los ejercicios se consideran datos sintéticos sobre salud mental. Contiene un archivo **bayesian.jl** y las visualizaciones se colocan en un directorio denominado output.

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


