from train_pca import train_pca
train_pca("healthy_takeoff.csv","pca_takeoff.pkl",3)
train_pca("healthy_cruise.csv","pca_cruise.pkl",3)
train_pca("healthy_descent.csv","pca_descent.pkl",3)