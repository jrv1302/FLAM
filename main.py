import numpy as np
import pandas as pd

def curve_points(theta_deg, M, X, n=4000, t_min=6, t_max=60):
    theta = np.radians(theta_deg)
    t = np.linspace(t_min, t_max, n)

    exp_term = np.exp(M * np.abs(t)) * np.sin(0.3 * t)

    x = t * np.cos(theta) - exp_term * np.sin(theta) + X
    y = 42 + t * np.sin(theta) + exp_term * np.cos(theta)

    return np.column_stack([x, y])

csv_path = "xy_data.csv"

df = pd.read_csv(csv_path)
data = df[["x", "y"]].values
pts = curve_points(30,0.03,55)
print(pts[:5])