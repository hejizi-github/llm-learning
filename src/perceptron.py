import numpy as np


class Perceptron:
    """感知机 — 从零实现，不使用任何 ML 库。
    与 nodes/01-perceptron-1958/perceptron.ipynb 内联实现完全一致。
    """

    def __init__(self, n_features, learning_rate=0.1):
        # 权重初始化为全 0（也可以随机，效果一样）
        self.w = np.zeros(n_features)
        self.bias = 0.0           # 偏置，相当于阈值的反面
        self.lr = learning_rate
        self.history = []         # 记录每轮错误数

    def predict(self, x):
        """给一个输入 x，返回 0 或 1。"""
        weighted_sum = np.dot(x, self.w) + self.bias
        # 超过 0 就输出 1，否则输出 0
        return 1 if weighted_sum > 0 else 0

    def train_one_epoch(self, X, y):
        """对所有数据跑一轮，更新权重。返回本轮错误数。"""
        errors = 0
        for xi, yi in zip(X, y):
            y_hat = self.predict(xi)   # 我猜的答案
            if y_hat != yi:            # 猜错了
                delta = self.lr * (yi - y_hat)
                self.w    += delta * xi   # 调整权重
                self.bias += delta        # 调整偏置
                errors += 1
        return errors

    def fit(self, X, y, max_epochs=100):
        """训练到收敛或达到最大轮数。供测试和快捷使用。"""
        self.history = []
        for epoch in range(max_epochs):
            errors = self.train_one_epoch(X, y)
            self.history.append(errors)
            if errors == 0:
                return self
        return self

    def accuracy(self, X, y):
        """计算在数据集上的准确率。"""
        correct = sum(self.predict(xi) == yi for xi, yi in zip(X, y))
        return correct / len(y)
