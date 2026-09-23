## 用途

本项目是「大气污染扩散反演与溯源平台」的代码仓库，用于逐步实现该方向的建模与数据处理能力。

## 环境与安装

- Python 3.11 及以上

```bash
python -m pip install -e .
```

## 测试

```bash
python -m pytest
```

基线尚无测试用例，收集到 0 个用例属预期结果。

## 命令行入口

安装后提供 `air-quality-modeling` 命令：

```bash
air-quality-modeling version    # 打印版本号
air-quality-modeling --help     # 打印用法
```

## 现有公开接口

- 命令行程序 `air-quality-modeling`
- Python 包 `air_quality`，其 `__version__` 为当前版本号
- 子模块 `air_quality.health`（健康影响评估，不入包根）：
  - `assess(C, U, P, beta, baseline=0)`：相对基线情景的健康影响评估
  - `aggregate(H, W, weights=None, z=1.96)`：跨情景加权聚合健康影响
  - `aggregate_correlated(H, F, weights=None, z=1.96)`：考虑受体误差相关的聚合
  - `sensitivity(H, W, weights=None, z=1.96)`：聚合结果的逐情景贡献分解，返回 `(M, R, C, S)`，其中 `M[i] = Σ(w[k]·H[k][i])`、`R[i] = z·sqrt(V[i])`、`C[k][i] = w[k]·(H[k][i] − M[i])`、`S[k][i] = w[k]·W[k][i]²/V[i]`（`V[i] = 0` 时 `S[k][i] = 0.0`）
  - `exceedance_probability(H, W, thresholds)`：情景总量超阈值概率
  - `risk_probability(H, W, thresholds, weights=None)`：跨情景加权超阈值风险
  - `risk_interval(H, W, thresholds, weights=None, z=1.96)`：聚合区间与超阈值风险
  - `level_probability(H, W, thresholds, weights=None)`：跨情景健康等级概率
  - `receptor_level_probability(H, W, thresholds)`：逐受体健康等级概率
  - `quantile(H, W, quantiles, weights=None, z=1.96)`：情景总量加权分位数
  - `receptor_quantile(H, W, quantiles, weights=None, z=1.96)`：逐受体加权分位数
