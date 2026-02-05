import pandas as pd
WINDOW=50
def generate_features(data):
    features=pd.DataFrame()
    for sensor in data.columns:
        features[f'{sensor}_mean'] = data[sensor].rolling(window=WINDOW).mean()
        features[f'{sensor}_std'] = data[sensor].rolling(window=WINDOW).std()
        
    return features.dropna()
