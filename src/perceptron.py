import numpy as np


class Perceptron:
    """感知机 — 从零实现，不使用任何 ML 库"""

    def __init__(self, learning_rate=0.1, max_epochs=100):
        self.lr = learning_rate
        self.max_epochs = max_epochs
        self.weights = None
        self.bias = None
        self.history = []  # 记录每轮的错误数

    def _step(self, z):
        """阶跃函数：z >= 0 返回 1，否则返回 0"""
        return (z >= 0).astype(int)

    def predict(self, X):
        """预测：z = X·w + b，然后用阶跃函数变成 0/1"""
        X = np.atleast_2d(X)
        z = X @ self.weights + self.bias  # 矩阵乘法 = 加权求和
        return self._step(z)

    def fit(self, X, y):
        """训练：感知机学习规则"""
        n_samples, n_features = X.shape
        # 初始化权重为 0
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.history = []

        for epoch in range(self.max_epochs):
            errors = 0
            for xi, yi in zip(X, y):
                pred = self.predict(xi)[0]
                error = int(yi) - int(pred)      # 误差：真相 - 预测
                if error != 0:
                    # 更新权重：w = w + lr × error × x
                    self.weights += self.lr * error * xi
                    self.bias    += self.lr * error
                    errors += 1
            self.history.append(errors)
            if errors == 0:
                print(f'收敛！第 {epoch + 1} 轮，错误数降到 0')
                return self

        print(f'达到最大轮数 {self.max_epochs}，停止训练（可能未收敛）')
        return self

    def accuracy(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)
