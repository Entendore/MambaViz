import pandas as pd
import numpy as np
from datetime import timedelta

def generate_complex_timeseries():
    np.random.seed(42)
    # 3 years of hourly data
    dates = pd.date_range(start='2020-01-01', periods=3*365*24, freq='H')
    t = np.arange(len(dates))
    
    # Trend
    trend = t * 0.005
    
    # Seasonalities (Daily, Weekly, Yearly)
    hour_of_day = dates.hour
    day_of_week = dates.dayofweek
    day_of_year = dates.dayofyear
    
    daily_season = 15 * np.sin(2 * np.pi * hour_of_day / 24)
    weekly_season = 10 * np.sin(2 * np.pi * day_of_week / 7)
    yearly_season = 30 * np.sin(2 * np.pi * day_of_year / 365)
    
    # Exogenous variables
    temp = 20 + 15 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 2, len(dates))
    humidity = 50 + 20 * np.cos(2 * np.pi * day_of_year / 365) + np.random.normal(0, 5, len(dates))
    
    # Random holidays/spikes
    spikes = np.zeros(len(dates))
    spike_days = np.random.choice(len(dates), 15, replace=False)
    for d in spike_days:
        spikes[d:d+24] = np.random.uniform(40, 80)
        
    # Target variable (e.g., Energy Demand)
    target = 50 + trend + daily_season + weekly_season + yearly_season + 0.5 * temp + spikes + np.random.normal(0, 3, len(dates))
    
    df = pd.DataFrame({
        'Date': dates,
        'Temperature': temp,
        'Humidity': humidity,
        'Is_Holiday_Spike': spikes,
        'Target': target
    })
    
    df.to_csv('timeseries_data.csv', index=False)
    print("✅ Generated 'timeseries_data.csv' with shape:", df.shape)

if __name__ == "__main__":
    generate_complex_timeseries()