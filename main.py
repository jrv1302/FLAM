import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.optimize import differential_evolution

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

csv_path = "xy_data.csv"

df = pd.read_csv(csv_path)
data = df[["x", "y"]].values
coarse_loss = make_loss(data,2000)

result = differential_evolution(
    coarse_loss,
    ((0,50),(-0.05,0.05),(0,100)),
    maxiter=300,
    popsize=40,
    seed=1,
    polish=True
)

print(result.x)