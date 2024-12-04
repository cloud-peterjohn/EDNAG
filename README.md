# DiffEvo
### 去噪公式
高适应度值对应着高概率密度，通过选择$g(x)$映射函数，将适应度函数$f(x)$映射到概率密度。
$$
p(x)=g[f(x)]
$$
在扩散模型的逆向去噪过程中，原始样本$x_0$即为期望获得高适应度值的个体。
$$
p(x_0=x)=g[f(x)] \tag{1}
$$
根据DDIM的公式，逆向扩散的去噪过程如下。为了实现去噪，需要求出$\hat{x_0}$和$\hat{\epsilon}$。
$$
x_{t-1} = \sqrt{\alpha_{t-1}}\cdot \hat{x_0} + \sqrt{1 - \alpha_{t-1} - \sigma^2_t}\cdot \hat{\epsilon} + \sigma_t w
$$

### 求出$\hat{x_0}$
扩散模型在每个时间步$t$，需要直接预测种群样本的原始样本($t=t_0$时的样本$\hat{x_0}$)，预测的方式需要通过当前时间步的样本$x_t$和适应度值$f(x_t)$得到。
$$
p(x_0=x|x_t)=\frac{p(x_0=x)\cdot p(x_t|x_0=x)}{p(x_t)}
$$
带入公式(1)。
$$
p(x_0=x|x_t)=\frac{g[f(x)]\cdot p(x_t|x_0=x)}{p(x_t)} \tag{2}
$$
因为扩散的过程是根据$x_0$加噪得到$x_t$，所以。
$$
p(x_t|x_0=x)\sim N(x_t; \sqrt{\alpha_t}x_{t-1}, 1 - \alpha_t)
$$
带入公式2。
$$
\hat{x_0}=\sum_{individual\_x\in populations}{x\cdot p(x_0=x|x_t)} \tag{?怎么变的?}
$$
$$
\hat{x_0}=\sum_{x\in pop}{x\cdot \frac{g[f(x)]\cdot N(x_t; \sqrt{\alpha_t}x_{t-1}, 1 - \alpha_t)}{p(x_t)}} \tag{3}
$$

### 求出$\hat{\epsilon}$
由于加噪的扩散过程为。
$$
x_0 = \frac{x_t - \sqrt{1 - \alpha_t }\cdot\epsilon}{\sqrt{\alpha_t}}
$$
所以，使用估计$\hat{x_0}$的求出$\hat{\epsilon}$。
$$
\hat{\epsilon} = \frac{x_t - \sqrt{\alpha_t}\cdot \hat{x_0}}{\sqrt{1 - \alpha_t}}
$$
$\hat{x_0}$在公式3中已经求出，带入即可。
所以，去噪过程的公式为。
$$
x_{t-1} = \sqrt{\alpha_{t-1}}\cdot \sum_{x\in pop}{x\cdot \frac{g[f(x)]\cdot N(x_t; \sqrt{\alpha_t}x_{t-1}, 1 - \alpha_t)}{p(x_t)}}  + \sqrt{1 - \alpha_{t-1} - \sigma^2_t}\cdot \frac{x_t - \sqrt{\alpha_t}\cdot \sum_{x\in pop}{x\cdot \frac{g[f(x)]\cdot N(x_t; \sqrt{\alpha_t}x_{t-1}, 1 - \alpha_t)}{p(x_t)}}}{\sqrt{1 - \alpha_t}} + \sigma_t w
$$

### 代码
在代码中，一步一步循环去噪迭代。
``` python
scheduler = DDIMSchedulerCosine(num_step=100)
for t, alpha in scheduler:
    fitness = two_peak_density(x, std=0.25)
    print(f"fitness {fitness.mean().item()}")
    # apply the power mapping function
    generator = BayesianGenerator(x, mapping_fn(fitness), alpha)
    x = generator(noise=0.1)
    trace.append(x)
```
通过estimator函数获取预测的$\hat{x_0}$。
``` python
def generate(self, noise=1.0, return_x0=False):
    x0_est = self.estimator(self.x)
    x_next = ddim_step(self.x, x0_est, (self.alpha, self.alpha_past), noise=noise)
    if return_x0:
        return x_next, x0_est
    else:
        return x_next
```
通过实现公式$x_{t-1} = \sqrt{\alpha_{t-1}}\cdot \hat{x_0} + \sqrt{1 - \alpha_{t-1} - \sigma^2_t}\cdot \hat{\epsilon} + \sigma_t w$来进行迭代去噪。
``` python
def ddim_step(xt, x0, alphas: tuple, noise: float = None):
    alphat, alphatp = alphas
    sigma = ddpm_sigma(alphat, alphatp) * noise
    eps = (xt - (alphat ** 0.5) * x0) / (1.0 - alphat) ** 0.5
    if sigma is None:
        sigma = ddpm_sigma(alphat, alphatp)
    x_next = (alphatp ** 0.5) * x0 + ((1 - alphatp - sigma ** 2) ** 0.5) * eps + sigma * torch.randn_like(x0)
    return x_next
```
$\sigma$通过$x_t$的方差$\sigma = \sqrt{\frac{(1 - \alpha_{t'})}{(1 - \alpha_t)} \cdot \left(1 - \frac{\alpha_t}{\alpha_t}\right)}$来计算，作者指出这是使用的DDPM默认公式。
``` python
def ddpm_sigma(alphat, alphatp):
    return ((1 - alphatp) / (1 - alphat) * (1 - alphat / alphatp)) ** 0.5
```

### 特点
- 不需要再训练神经网络来求得$\hat{\epsilon}$。
- 不需要专门的条件嵌入就可以实现向高适应度值区域定向去噪。
- 解空间范围广，不局限于生成模型的子空间，更接近全局最优。

### 收敛性与多样性的平衡
```elite_rate```保证了最佳样本能够在迭代中保留下来，确保了收敛性，```diver_rate```保证了差异性大的样本能够在迭代中保留下来，确保了多样性。  
```mutate_rate```控制发生变异的样本数，```mutate_distri_index```控制变异的规模尺度，确保了对解空间的探索性。


# EmoDM
EmoDM认为扩散模型的正向加噪过程是逆向的进化，逆向去噪过程是正向的进化。他们实现了无需进化的多目标优化方法，以去噪替代进化。EmoDM的条件嵌入方式在于注意力机制，每次去噪后，都会采用注意力机制，让种群乘以注意力矩阵，而注意力矩阵并非是一个可训练参数，而是由每个个体$x_t^i$和目标$y_t^i=\{f_1(x_t^i),f_1(x_t^i),\ldots,f_m(x_t^i)\}$通过相互熵计算得到的，$i\in\{0,1,\ldots,N\}$，其中，$N$为种群中个体数量，$m$为目标的数量。
互信息熵的计算如公式4。
$$
I(x_t; y_t) = \sum_{x_t} \sum_{y_t} p(x_t, y_t) \log \left( \frac{p(x_t, y_t)}{p(x_t) \cdot p(y_t)} \right) \tag{4}
$$
其中，$p(x_t)$和$p(y_t)$分别为决策变量和目标变量的边缘概率密度，$p(x_t, y_t)$为联合概率密度，均由$(x,y)$统计得到。
如果互信息熵为零，则表示该决策变量$x_t^i$的观测值不提供任何关于目标$y_t$的附加信息。较大的互信息熵可能表明决策变量$x_t^i$对目标$y_t$的影响更加复杂或密切。
$$
x^i=x^i \cdot softmax(\frac{1}{m} \cdot \sum_{j=1}^{m}{I(x_t^i; y_t^j)}) \tag{5}
$$
总共m个目标通过求平均，得到NMI(Normalized Mutual Information)。通过公式5，在每一次去噪迭代循环中，更新$x^i$的值，在目标上表现更佳的个体将被注意力机制赋予更大的权重，表现更差的个体将被注意力机制赋予更小的权重，以实现向多个目标进化。

# EvoDiff-MSA
## Motivation
现有SOTA方法，限制了训练数据的范围(?)，且生成的蛋白质结构限制在设计空间的一个小的且有偏差的子集上。
将DDPM的前向扩散过程设计为Mutate或加Mask，逆向去噪过程为隐式条件嵌入。