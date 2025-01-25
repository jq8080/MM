import numpy as np
from scipy.optimize import minimize

# 定义对数似然函数和梯度
def log_likelihood(beta, X, rankings):
    theta = np.exp(X @ beta)
    log_l = 0.0
    for rank in rankings:
        remaining = list(range(len(theta)))
        for pos in rank:
            log_l += np.log(theta[pos]) - np.log(theta[remaining].sum())
            remaining.remove(pos)
    return -log_l  # 负对数似然（因scipy最小化）

def gradient(beta, X, rankings):
    grad = np.zeros_like(beta)
    theta = np.exp(X @ beta)
    for rank in rankings:
        remaining = list(range(len(theta)))
        for pos in rank:
            # 计算梯度项
            grad += X[pos] - (X[remaining].T * theta[remaining]).sum(axis=1) / theta[remaining].sum()
            remaining.remove(pos)
    return -grad  # 负梯度

# 计算预测排名
def predict_ranking(beta, X):
    theta = np.exp(X @ beta)
    return np.argsort(theta)  # 从大到小排序

# 计算准确性
def calculate_accuracy(predicted_rankings, true_rankings):
    correct = 0
    total = 0
    for pred, true in zip(predicted_rankings, true_rankings):
        if np.array_equal(pred, true):
            correct += 1
        total += 1
    return correct / total

# 示例数据
X = np.array([[1.2, 1, 0.3], [0.8, 0, 0.5], [-0.5, 0, 0.8], [-1.0, 0, 0.6]])
rankings = [[0, 1, 2, 3]]  # A > B > C > D

# 初始参数
beta0 = np.zeros(3)

# BFGS优化
result = minimize(log_likelihood, beta0, args=(X, rankings), jac=gradient, method='BFGS')
print("估计参数:", result.x)

# 预测排名
predicted_rankings = [predict_ranking(result.x, X)]
print("预测排名:", predicted_rankings)

# 计算准确性
accuracy = calculate_accuracy(predicted_rankings, rankings)
print("模型准确性:", accuracy)