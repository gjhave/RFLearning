"""Robbins-Monro theorem is an pioneering work in the field of stochastic approximation"""

import numpy as np
import tqdm

import matplotlib.pyplot as plt


def ImportanceSampling():
    """演示重要性采样"""
    import torch
    xset = [i for i in range(100)]
    p0 = np.random.normal(2, 1, size = len(xset))
    p0 = torch.softmax(torch.tensor(p0), dim=0).numpy()
    p1 = np.random.normal(5, 1, size = len(xset))
    p1 = torch.softmax(torch.tensor(p1), dim=0).numpy()

    s0 = np.random.choice(xset, p=p0, size=10000)
    m0 = np.mean(s0)
    s1 = np.random.choice(xset, p=p1, size=10000)
    m1 = np.mean(s1)
    print(m1, m0)

    plt.plot(xset, p0, label="p0")
    plt.plot(xset, p1, label="p1")
    plt.legend()
    plt.show()

ImportanceSampling()




def MeanEstimation():
    """
    w_{k+1} = w_{k} - (w_{k} - x_{k})/k
    w_{k}为第k次估计，x_{k}为第k次采样
    """
    xset = np.random.randint(0, 100, size=(10000,))
    xmean = float(np.mean(xset))  # 计算数据的真实均值
    error = np.random.normal(0, 1, size=(100000,))  # 模拟噪声，均值为0，标准差为1

    # w - x = [w - E(x)] + [E(x) - x] = g(w) + error
    def g(w):  # 这样定义符合RM算法的表示
        return w - np.random.choice(xset)

    w_iter = []
    for i in range(10000):
        if i == 0:
            w = np.random.choice(xset)  # 随机初始化一个w
        else:
            w = (
                w - (g(w) + error[i]) / i
            )  # 更新估计值，目标是使w接近数据的真实均值,这里可以将error设为0
        w_iter.append(w)

    plt.plot(w_iter)
    plt.axhline(xmean, color="r", linestyle="--", label="Target Value")
    plt.title("Convergence of SGD Estimate")
    plt.xlabel("Iteration")
    plt.ylabel("Estimated Mean")
    plt.legend()
    plt.show()


def RobbinsMonro():
    """
    RM理论成立条件：
    1. 目标函数f(w)在w=theta处有唯一根，即f(theta)=0，f(w)的倒数存在且连续0<c_{1}<=f'(w)<=c_{2}，即f(w)是单调的；
    2. 学习率α_{k}满足sum(α_{k})=inf且sum((α_{k})^2)<inf；即学习率逐渐减小但不太快。
    3. 噪声项epsilon_n满足E[epsilon_n|w_n]=0且Var(epsilon_n|w_n)<=sigma^2，即噪声是零均值且方差有界的随机变量。
    在满足上述条件下，RM算法的迭代过程w_{n+1}=w_n - alpha_n * (f(w_n) + epsilon_n)将以概率1收敛到theta，即w_n -> theta as n->inf。
    该算法的核心思想是通过不断调整参数w来逼近目标函数f的根theta，学习率alpha_n控制了每次更新的步长，而噪声项epsilon_n则模拟了实际应用中可能存在的随机扰动。
    """
    """
    求解g(x)=0的根x_{*}，迭代方法，即RM算法：
    x_{k+1} = x_{k} - α_{k}*g~(x_{k}, error_{k})
    这里g~=g(x)+error：真实值加上一个误差，即观测值
    """
    error = np.random.normal(0, 1, size=(100000,))  # 模拟噪声，均值为0，标准差为1

    def g(x):  # 这里定义一个g(x)函数，要求g(x)=0的根
        # return np.power(x, 3) - 5
        return np.tanh(x - 1.71)

    x_iter = []
    x = 0
    for i in range(10000):
        if i == 0:
            x = 2  # 模拟真实输出
        else:
            x = x - (g(x) + error[i]) / i
            # 这里 g(x)+error表示带有误差的观测值，即g~(x_{k}, erro_{k}),实际问题中，我们可能不知道g(x)的表达式，只能输入一个值，获取一个带噪声的观测结果。这里演示所以给定了g(x)
        x_iter.append(x)

    plt.plot(x_iter)
    plt.axhline(1.71, color="r", linestyle="--", label="Target Value 1.71")
    plt.title("Convergence of SGD Estimate")
    plt.xlabel("Iteration")
    plt.ylabel("Estimated Mean")
    plt.legend()
    plt.show()


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


def ContractMap():
    """收缩映射"""

    def func(x):
        return 0.5 * np.sin(
            x
        )  # 定义一个简单的函数，目标是找到其不动点，即x = func(x)的解

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


ContractMap()
