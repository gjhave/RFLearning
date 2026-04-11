"""Robbins-Monro theorem is an pioneering work in the field of stochastic approximation"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def ContractionMapping():
    """收缩映射"""
    # 定义一个简单的函数，目标是找到其不动点，即x = func(x)的解
    def func(x):
        return 0.5 * np.sin(x)  

    x_iter = []
    x = 2  # 初始值
    for i in range(20):
        x = func(x)  # 更新x，目标是使x接近func(x)，即找到不动点
        x_iter.append(x)
    plt.plot(x_iter)
    plt.axhline(0, color="r", linestyle="--", label="Target Value")
    plt.title("Convergence of Contraction Mapping")
    plt.xlabel("Iteration")
    plt.ylabel("Estimated Fixed Point")
    plt.legend()
    plt.show()

# ContractionMapping()

def RobbinsMonro():
    """
    求解g(x)=0的根x_{*}，迭代方法，即RM算法：
    x_{k+1} = x_{k} - α_{k}*g~(x_{k}, error_{k})
    这里g~=g(x)+error：真实值加上一个误差，即观测值
    """
    def g_erro(x, erro): 
        return np.tanh(x - 1.71) + erro

    w_iter = []
    w = 0
    for i in range(100000):
        if i == 0:
            w = 2  # 初始化一个值
        else:
            error = np.random.normal(0, 1)
            w = w - 0.0001 * g_erro(w, error)
        w_iter.append(w)

    plt.plot(w_iter)
    plt.axhline(1.71, color="r", linestyle="--", label="Target Value 1.71")
    plt.title("Convergence of SGD Estimate")
    plt.xlabel("Iteration")
    plt.ylabel("Estimated Mean")
    plt.legend()
    plt.show()
# RobbinsMonro()


def MeanEstimation():
    """
    w_{k+1} = w_{k} - (w_{k} - x_{k})/k
    w_{k}为第k次估计，x_{k}为第k次采样
    """
    xset = np.random.randint(0, 100, size=(10000,))
    xmean = float(np.mean(xset))  # 计算数据的真实均值

    # g(w) = w - E[x]
    def g_erro(w, error):  # 这样定义符合RM算法的表示
        return w - np.random.choice(xset) + error

    w_iter = []
    for i in range(1000000):
        if i == 0:
            w = np.random.choice(xset)  # 随机初始化一个w
        else:
            error = np.random.normal(0, 1)
            w = w - 0.0001 * g_erro(w, error)
        w_iter.append(w)

    plt.plot(w_iter)
    plt.axhline(xmean, color="r", linestyle="--", label="Target Value")
    plt.title("Convergence Mean Estimation")
    plt.xlabel("Iteration")
    plt.ylabel("Estimated Mean")
    plt.legend()
    plt.show()
# MeanEstimation()

def GradientDescent():
    """
    梯度下降算法实际上是RM算法的应用，假设我们要找到某个函数f(w, X)的最优解参数w，实际上就是要让他的导数f'_{w}(w, X)=0，即RM算法去中g(x)=f'_{w}(w, X)=0
    w_{k+1} = w_{k} - α_{k}*E[f'_{w}(w_{k}, X)]
    """
    xset = np.random.randint(-100, 500, size=(1000,)) / 100  # 模拟输入数据

    # 对于f(w，X) = 0.5 * (w - X)^2的导数是(w - x)
    def gradientfunc(w, xset):
        return (
            w - xset
        )  # 计算梯度，即f'(w) = (w - x) ，目标是使w接近x的均值，即w -> mean(xset) as n->inf。

    w_iter = []
    w = 3  # 初始值
    for i in range(10000):
        w = w - 0.001 * np.mean(gradientfunc(w, xset))  # 更新参数，学习率为0.1
        w_iter.append(w)

    plt.plot(w_iter)
    plt.axhline(float(np.mean(xset)), color="r", linestyle="--", label="Target Value")
    plt.title("Convergence of Gradient Descent")
    plt.xlabel("Iteration")
    plt.ylabel("Estimated Minimum")
    plt.legend()
    plt.show()


def SGD():
    """
    随机梯度下降，是把梯度下降算法中，每个迭代中都要使用所有的X，变成了每次只采样一个x，因此下面的表达是中就没有expectation, X变成了x_{k}
    w_{k+1} = w_{k} - α_{k}*f'_{w}(w_{k}, x_{k})
    """
    xset = np.random.randint(-100, 100, size=(1000,)) / 100  # 模拟输入数据

    # 对于f(w，X) = 0.5 * (w - X)^2的导数是(w - x)
    # 计算梯度，即f'(w) = (w - x) 对于f(w) = 0.5 * (w - x)^2的情况，目标是使w接近x的均值，即w -> mean(xset) as n->inf。
    def gradientfunc(w, x):
        return w - x

    w_iter = []
    w = 3  # 初始值
    for i in range(10000):
        x = np.random.choice(xset)  # 随机选择一个样本进行更新
        # w = w - α*(w - x)也是RM算法的形式
        w = w - 0.001 * gradientfunc(w, x)  # 更新参数，学习率为0.1
        w_iter.append(w)

    plt.plot(w_iter)
    plt.axhline(float(np.mean(xset)), color="r", linestyle="--", label="Target Value")
    plt.title("Convergence of Stochastic Gradient Descent")
    plt.xlabel("Iteration")
    plt.ylabel("Estimated Minimum")
    plt.legend()
    plt.show()


def MBGD(batch_size=10):
    """
    batch_size=1,即每次更新使用一个样本，等价于SGD；
    batch_size=n，即每次更新使用全部样本，等价于GD；
    0<batch_size<n，即每次更新使用一个小批量样本，介于SGD和GD之间，通常能更快收敛且更稳定。
    """
    xset = np.random.randint(-100, 100, size=(1000,)) / 100  # 模拟输入数据

    def gradientfunc(w, x):
        return np.tanh(
            w - x
        )  # 计算梯度，即f'(w) = (w - x) 对于f(w) = 0.5 * (w - x)^2的情况，目标是使w接近x的均值，即w -> mean(xset) as n->inf。

    w_iter = []
    w = 1  # 初始值
    for i in range(10000):
        batch = np.random.choice(
            xset, size=batch_size, replace=False
        )  # 随机选择一个小批量样本进行更新
        w = w - 0.001 * np.mean(gradientfunc(w, batch))  # 更新参数，学习率为0.1
        w_iter.append(w)

    plt.plot(w_iter)
    plt.axhline(float(np.mean(xset)), color="r", linestyle="--", label="Target Value")
    plt.title("Convergence of Mini-Batch Gradient Descent")
    plt.xlabel("Iteration")
    plt.ylabel("Estimated Minimum")
    plt.legend()
    plt.show()


def basic_importance_sampling():
    """
    目标：估计 E_{X~N(0,1)}[f(X)]，其中 f(x) = max(0, x) 的期望
    使用重要性采样从 N(2,1) 采样来估计
    """
    
    np.random.seed(42)
    n_samples = 1000
    
    # 目标分布 p(x)：标准正态分布 N(0,1)
    # 提议分布 q(x)：N(2,1) - 偏移的正态分布
    def target_pdf(x):
        return stats.norm.pdf(x, loc=0, scale=1)
    
    def proposal_pdf(x):
        return stats.norm.pdf(x, loc=2, scale=1)
    
    # 要估计的函数 f(x)
    def f(x):
        return np.maximum(0, x)  # ReLU 函数
    
    # 从提议分布采样
    samples = np.random.normal(loc=2, scale=1, size=n_samples)
    
    # 计算重要性权重
    weights = target_pdf(samples) / proposal_pdf(samples)
    
    # 重要性采样估计
    f_values = f(samples)
    estimate = np.sum(weights * f_values) / np.sum(weights)
    
    # 真实值（解析解或蒙特卡洛）
    true_value = 1 / np.sqrt(2 * np.pi)  # E[max(0, X)] for X~N(0,1)
    
    print("=" * 50)
    print("基础重要性采样示例")
    print("=" * 50)
    print(f"目标分布: N(0,1), 提议分布: N(2,1)")
    print(f"函数 f(x) = max(0, x)")
    print(f"重要性采样估计: {estimate:.4f}")
    print(f"真实值: {true_value:.4f}")
    print(f"相对误差: {abs(estimate - true_value)/true_value*100:.2f}%")
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 左图：分布对比
    x_range = np.linspace(-4, 6, 200)
    ax1 = axes[0]
    ax1.plot(x_range, target_pdf(x_range), 'b-', label='Target p(x): N(0,1)', linewidth=2)
    ax1.plot(x_range, proposal_pdf(x_range), 'r--', label='Proposal q(x): N(2,1)', linewidth=2)
    ax1.hist(samples, bins=30, density=True, alpha=0.3, color='red', label='Samples from q')
    ax1.set_xlabel('x')
    ax1.set_ylabel('Density')
    ax1.set_title('分布对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右图：权重分布
    ax2 = axes[1]
    ax2.hist(weights, bins=50, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Importance Weights')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'权重分布 (有效样本量: {1/np.sum((weights/np.sum(weights))**2):.0f}/{n_samples})')
    ax2.axvline(1, color='red', linestyle='--', label='Weight=1')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return estimate

basic_importance_sampling()
