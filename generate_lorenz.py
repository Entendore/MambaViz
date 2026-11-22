import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp

def lorenz_system(t, state, sigma=10, rho=28, beta=8/3):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

def generate_lorenz_data():
    np.random.seed(42)
    
    # Initial state
    initial_state = [1.0, 1.0, 1.0]
    
    # Time span (long enough to capture chaotic dynamics)
    t_span = (0, 50)
    t_eval = np.linspace(t_span[0], t_span[1], 10000)
    
    # Solve ODE
    sol = solve_ivp(lorenz_system, t_span, initial_state, t_eval=t_eval)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Time': sol.t,
        'X': sol.y[0],
        'Y': sol.y[1],
        'Z': sol.y[2]
    })
    
    # We want to predict the next X value
    df['Target'] = df['X'].shift(-1)
    df = df.dropna()
    
    df.to_csv('lorenz_data.csv', index=False)
    print("✅ Generated 'lorenz_data.csv' with shape:", df.shape)

if __name__ == "__main__":
    generate_lorenz_data()