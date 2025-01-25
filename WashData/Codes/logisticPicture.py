import numpy as np
import matplotlib.pyplot as plt

# 设置逻辑回归函数的参数
k = 1  # 曲线的陡峭程度
x_0 = 5  # 曲线的中点

# 生成参与次数（投入成本）数据
x = np.linspace(0, 10, 100)  # 参与次数从0到10

# 逻辑回归函数 f(x)
f_x = 1 / (1 + np.exp(-k * (x - x_0)))

# 逻辑回归的导数 f'(x)（即收益率）
f_prime_x = k * f_x * (1 - f_x)

# 假设三个国家的参与次数
A_participation = x + np.random.uniform(-1, 1, len(x))  # 国家A参与次数
B_participation = x + np.random.uniform(-1.5, 1.5, len(x))  # 国家B参与次数
C_participation = x + np.random.uniform(-0.5, 0.5, len(x))  # 国家C参与次数

# 计算每个国家的收益率（导数值）
A_efficiency = k * (1 / (1 + np.exp(-k * (A_participation - x_0)))) * (1 - (1 / (1 + np.exp(-k * (A_participation - x_0)))))
B_efficiency = k * (1 / (1 + np.exp(-k * (B_participation - x_0)))) * (1 - (1 / (1 + np.exp(-k * (B_participation - x_0)))))
C_efficiency = k * (1 / (1 + np.exp(-k * (C_participation - x_0)))) * (1 - (1 / (1 + np.exp(-k * (C_participation - x_0)))))

# 绘制图表
plt.figure(figsize=(10, 6))
plt.plot(x, A_efficiency, label='Country A', color='blue')
plt.plot(x, B_efficiency, label='Country B', color='green')
plt.plot(x, C_efficiency, label='Country C', color='red')

# 添加标题和标签
plt.title('Efficiency (Marginal Utility) vs. Participation (Investment Cost)', fontsize=14)
plt.xlabel('Participation (Investment Cost)', fontsize=12)
plt.ylabel('Efficiency (Marginal Utility)', fontsize=12)
plt.legend()

# 显示图表
plt.grid(True)
plt.show()
