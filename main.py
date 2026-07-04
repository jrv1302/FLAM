import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.optimize import differential_evolution, minimize

def curve_points(theta_deg, M, X, n=4000, t_min=6, t_max=60):
    theta = np.radians(theta_deg)
    t = np.linspace(t_min, t_max, n)

    exp_term = np.exp(M * np.abs(t)) * np.sin(0.3 * t)

    x = t * np.cos(theta) - exp_term * np.sin(theta) + X
    y = 42 + t * np.sin(theta) + exp_term * np.cos(theta)

    return np.column_stack([x, y])

def make_loss(data, n_samples):
    def loss(params):
        theta, M, X = params

        pts = curve_points(theta, M, X, n=n_samples)

        tree = cKDTree(pts)

        d, _ = tree.query(data, p=1)

        return np.mean(d)

    return loss

def fit(data, bounds=((0, 50), (-0.05, 0.05), (0, 100)), seed=1):
    coarse_loss = make_loss(data, n_samples=2000)
    result = differential_evolution(
        coarse_loss,
        bounds,
        maxiter=300,
        popsize=40,
        tol=1e-12,
        seed=seed,
        polish=True,
    )

    fine_loss = make_loss(data, n_samples=8000)
    refined = minimize(
        fine_loss,
        result.x,
        method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 5000},
    )

    return refined.x, refined.fun

csv_path = "xy_data.csv"

df = pd.read_csv(csv_path)
data = df[["x", "y"]].values

(theta_deg, M, X), final_loss = fit(data)
theta_rad = np.radians(theta_deg)

print("Fitted parameters:")
print(f"theta = {theta_deg:.6f} deg ({theta_rad:.6f} rad)")
print(f"M = {M:.6f}")
print(f"X = {X:.6f}")
print(f"mean L1 distance (fit loss) = {final_loss:.6f}")

pts = curve_points(theta_deg, M, X, n=20000)
tree = cKDTree(pts)
d, _ = tree.query(data, p=1)
print(f"\nL1 distance over all {len(data)} points:")
print(f"max L1 error = {d.max():.6f}")
print(f"mean L1 error = {d.mean():.6f}")
print(f"total L1 error = {d.sum():.6f}")