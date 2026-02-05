import pandas as pd
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from feature_engineering import generate_features

def train_pca(csv_path,model_path,n_components):
    df=pd.read_csv(csv_path)
    features=generate_features(df)
    scaler=StandardScaler()
    scaled_features=scaler.fit_transform(features)
    pca=PCA(n_components=n_components)
    pca.fit(scaled_features)
    scaled_recon=pca.inverse_transform(pca.transform(scaled_features))
    recon_error=((scaled_features-scaled_recon)**2).mean(axis=1)
    model={
        "scaler":scaler,
        "pca":pca,
        "recon_error":recon_error.mean(),
        "error_std":recon_error.std(),
        "error_threshold": np.percentile(recon_error, 99.5)

    }
    joblib.dump(model,model_path)
    print(f"PCA model trained and saved to {model_path}")


   