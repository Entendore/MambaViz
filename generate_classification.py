import pandas as pd
import numpy as np

def generate_classification_data():
    np.random.seed(42)
    n_samples = 5000
    seq_length = 20 # Each "sample" logically spans 20 steps
    
    # We generate a continuous stream of features and regimes
    total_length = n_samples + seq_length
    
    regimes = np.zeros(total_length)
    features = np.zeros((total_length, 3))
    
    current_regime = 0
    for i in range(total_length):
        # Randomly switch regime occasionally
        if np.random.rand() < 0.01:
            current_regime = np.random.choice([0, 1, 2])
        regimes[i] = current_regime
        
        # Generate features based on regime
        if current_regime == 0: # Calm waters
            features[i, 0] = np.random.normal(0, 0.5)
            features[i, 1] = np.random.normal(0, 0.5)
            features[i, 2] = np.random.normal(0, 0.5)
        elif current_regime == 1: # High volatility
            features[i, 0] = np.random.normal(0, 3.0)
            features[i, 1] = np.random.normal(0, 3.0)
            features[i, 2] = np.random.normal(0, 3.0)
        elif current_regime == 2: # Upward trend
            features[i, 0] = features[i-1, 0] + 0.2 + np.random.normal(0, 0.5)
            features[i, 1] = np.random.normal(1, 1.0)
            features[i, 2] = np.random.normal(0, 1.0)

    # Create DataFrame
    df = pd.DataFrame(features, columns=['Feature_1', 'Feature_2', 'Feature_3'])
    df['Target'] = regimes.astype(int)
    
    df.to_csv('classification_data.csv', index=False)
    print("✅ Generated 'classification_data.csv' with shape:", df.shape)

if __name__ == "__main__":
    generate_classification_data()