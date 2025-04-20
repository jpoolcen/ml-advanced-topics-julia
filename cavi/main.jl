# Ejemplo de uso para GMM
include("cavi.jl")
using .Cavi

K = 2
N = 300
mu_true = [-3.0, 3.0]
x, z_true = generate_data(N, K, mu_true)

r, m, s2 = initialize_variational(N, K)

r, m, s2,elbos = cavi_iteration(x, r, m, s2, max_iter=50)


show_cluster_assignments(r, x, z_true, save_path="clusters_cavi.png")

plot_elbo(elbos, save_path="elbo_evolution.png")

println("ELBO final: ", compute_elbo(x, r, m, s2))