class Kalman1D:
    def __init__(self, x0, Q=0.01, R=0.1):
        self.x = x0
        self.P = 1.0
        self.Q = Q
        self.R = R

    def predict(self):
        self.P += self.Q
        return self.x

    def update(self, z):
        K = self.P / (self.P + self.R)
        self.x += K * (z - self.x)
        self.P *= (1 - K)
        return self.x
