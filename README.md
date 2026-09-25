## 用途

本项目是「大气污染扩散反演与溯源平台」的代码仓库，用于该方向的建模与数据处理，目前已提供高斯烟羽扩散、情景健康影响评估与风险聚合等算法。

## 环境与安装

- Python 3.11 及以上

```bash
python -m pip install -e .
```

## 测试

```bash
python -m pytest
```

当前尚无测试用例，收集到 0 个用例属预期结果。

## 命令行入口

安装后提供 `air-quality-modeling` 命令：

```bash
air-quality-modeling version    # 打印版本号
air-quality-modeling --help     # 打印用法
```

## 现有公开接口

- 命令行程序 `air-quality-modeling`
- Python 包 `air_quality`，其 `__version__` 为当前版本号
- 包根公开函数：`persistence`、`forecast`、`forecast_interval`、`aggregate`
- 健康影响评估模块 `air_quality.health`（通过 `air_quality.health.xxx` 调用，不在包根导出），提供以下函数：
  - `assess(C, U, P, beta, baseline=0)`：评估各情景相对基线情景的健康影响，返回 `(H, S, W, Q)`
  - `aggregate(H, W, weights=None, z=1.96)`：跨情景聚合各受体健康影响及区间，返回 `(M, R, lower, upper, total, total_spread)`
  - `aggregate_correlated(H, F, weights=None, z=1.96)`：考虑受体误差相关时的跨情景聚合
  - `any_receptor_probability(H, W, thresholds, weights=None)`：假定同一情景内各受体误差独立时，至少一个受体超过各级阈值的跨情景加权概率，返回阈值序的 3 长 `list[float]`
  - `at_least_count_probability(H, W, thresholds, min_count=1, weights=None)`：假定同一情景内各受体误差独立时，至少 `min_count` 个受体超过各级阈值的跨情景加权概率；`min_count` 为非 bool 且 `>=1` 的 int（`>N` 时概率为 0），返回阈值序的 3 长 `list[float]`，不舍入
  - `exceedance_probability(H, W, thresholds)`：各情景总影响超过各级阈值的概率
  - `risk_probability(H, W, thresholds, weights=None)`：跨情景加权聚合的超标风险概率
  - `risk_interval(H, W, thresholds, weights=None, z=1.96)`：跨情景聚合影响区间与超标风险
  - `level_probability(H, W, thresholds, weights=None)`：跨情景加权聚合的健康等级概率
  - `receptor_level_probability(H, W, thresholds)`：分受体的健康等级概率
  - `receptor_level_interval(H, W, thresholds, weights=None, z=1.96)`：分受体期望健康等级的跨情景加权均值与区间。`H`、`W` 为同形 K×N（K,N>0）list/tuple 矩阵，元素为非 bool 有限数（`H` 可负，`W` 非负）；`thresholds` 恰为 3 个严格递增的非 bool 有限非负数；`weights` 为 `None`（取等权 `1/K`）或 K 个非负有限权且 `fsum>0`（按 `fsum` 归一）；`z` 为非负有限数。容器/元素类型错抛 `TypeError`；空、锯齿、形状/长度不符、非有限、负值、阈值顺序、权和或 `z` 非法抛 `ValueError`。对 `μ=H[k][i]`、`σ=W[k][i]`：`σ>0` 时 `qj=erfc((Tj−μ)/(σ*sqrt(2)))/2`，否则 `Tj≤μ` 取 `1.0`、否则 `0.0`；`p=[1−q0, q0−q1, q1−q2, q2]`，`e[k][i]=fsum(r*p[r] for r in 0..3)`、`v[k][i]=fsum(r²*p[r] for r in 0..3)−e[k][i]²`。`M[i]=fsum(w[k]*e[k][i])`、`V[i]=fsum(w[k]*((e[k][i]−M[i])²+v[k][i]))`、`R=z*sqrt(V)`、`L=M−R`、`U=M+R`，返回受体序 4 项 N 长 `list[float]` 的 `(mean, spread, lower, upper)`，均不舍入
  - `receptor_level_quantile(H, W, thresholds, quantiles, weights=None)`：分受体健康等级的跨情景加权分位数。`H`、`W` 为同形 K×N 矩阵（`H` 元素可负，`W` 元素非负）；`thresholds` 为 3 个严格递增非负数；`quantiles` 为非空、`[0,1]` 内非递减序列；`weights` 为 `None`（取等权 `1/K`）或按 `fsum` 归一的 K 个非负权。对 `σ=W[k][i]`：`σ>0` 时 `qj=erfc((thresholds[j]−H[k][i])/(σ*sqrt(2)))/2`，否则 `qj=1.0`（`thresholds[j]≤H[k][i]`）或 `0.0`；`p[k][i]=[1−q0, q0−q1, q1−q2, q2]`，`P[i][r]=fsum(w[k]*p[k][i][r])`；分位 `q=0` 取 `0`，否则取最小等级 `r` 使 `fsum(P[i][s] for s in 0..r)>=q`。返回分位×受体序的 M×N `list[list[int]]`
  - `receptor_expected_excess(H, W, thresholds, weights=None)`：分受体、跨情景加权的超标概率与期望超额，返回 N×3 的 `(probabilities, excess)`
  - `receptor_count_interval(H, W, thresholds, weights=None, z=1.96)`：假定同一情景内各受体误差独立时，超过各级阈值的受体数的跨情景加权均值与区间，返回阈值序 3 长 `list[float]` 的 `(mean, spread, lower, upper)`，其中 `lower=max(0, mean-spread)`、`upper=min(N, mean+spread)`，均不舍入
  - `receptor_level_count_interval(H, W, thresholds, weights=None, z=1.96)`：假定同一情景内各受体误差独立时，处于各健康等级的受体数的跨情景加权均值与区间。`H`、`W` 为同形 K×N（K,N>0）list/tuple 矩阵，元素为非 bool 有限数（`H` 可负，`W` 非负）；`thresholds` 恰为 3 个严格递增的非 bool 有限非负数；`weights` 为 `None`（取等权 `1/K`）或 K 个非负有限权且 `fsum>0`（按 `fsum` 归一）；`z` 为非负有限数。容器/元素类型错抛 `TypeError`；空、锯齿、形状/长度不符、非有限、负值、阈值顺序、权和或 `z` 非法抛 `ValueError`。对 `μ=H[k][i]`、`σ=W[k][i]`：`σ>0` 时 `q[k][i][j]=erfc((thresholds[j]−μ)/(σ*sqrt(2)))/2`，否则 `thresholds[j]≤μ` 取 `1.0`、否则 `0.0`；`p[k][i]=[1−q0, q0−q1, q1−q2, q2]`，`c[k][r]=fsum(p[k][i][r] for i in 0..N−1)`、`v[k][r]=fsum(p[k][i][r]*(1−p[k][i][r]) for i in 0..N−1)`。`M[r]=fsum(w[k]*c[k][r])`、`V[r]=fsum(w[k]*((c[k][r]−M[r])²+v[k][r]))`、`R=z*sqrt(V)`、`L=max(0,M−R)`、`U=min(N,M+R)`，返回等级 0..3 序 4 项 4 长 `list[float]` 的 `(mean, spread, lower, upper)`，均不舍入
  - `receptor_level_count_quantile(H, W, thresholds, quantiles, weights=None)`：假定同一情景内各受体误差独立时，处于各健康等级的受体数的跨情景加权分位数（一热向量形式）。`H`、`W` 为同形 K×N（K,N>0）list/tuple 矩阵，元素为非 bool 有限数（`H` 可负，`W` 非负）；`thresholds` 恰为 3 个严格递增的非 bool 有限非负数；`quantiles` 为非空、`[0,1]` 内非递减的非 bool 有限序列；`weights` 为 `None`（取等权 `1/K`）或 K 个非负有限权且 `fsum>0`（按 `fsum` 归一）。容器/元素类型错抛 `TypeError`；空、锯齿、形状/长度不符、非有限、负值、阈值/分位顺序或权和非法抛 `ValueError`。对 `μ=H[k][i]`、`σ=W[k][i]`：`σ>0` 时 `qj=erfc((thresholds[j]−μ)/(σ*sqrt(2)))/2`，否则 `qj=1.0`（`thresholds[j]≤μ`）或 `0.0`；`p[k][i]=[1−q0, q0−q1, q1−q2, q2]`；`d[k][l]` 自 `[1]+[0]*N` 起按 `d′[r]=d[r](1−p[k][i][l])+(r>0?d[r−1]p[k][i][l]:0)` 逐个折入受体，`P[l][r]=fsum(w[k]*d[k][l][r])`。`R[m][l]` 为 N+1 长 int 一热向量：分位 `q=0` 取 `r=0`，否则取最小 `r` 使 `fsum(P[l][0..r])>=q`，该位置为 1 其余为 0。返回分位×等级×计数序的 M×4×(N+1) `list[list[list[int]]]`，按序不舍入
  - `receptor_level_count_share(H, W, thresholds, weights=None)`：假定同一情景内各受体误差独立时，各情景对“处于各健康等级的受体数”分布的概率占比与期望计数占比。`H`、`W` 为同形 K×N（K,N>0）list/tuple 矩阵，元素为非 bool 有限数（`H` 可负，`W` 非负）；`thresholds` 恰为 3 个严格递增的非 bool 有限非负数；`weights` 为 `None`（取等权 `1/K`）或 K 个非负有限权且 `fsum>0`（按 `fsum` 归一）。容器/元素类型错抛 `TypeError`；空、锯齿、形状/长度不符、非有限、负值、阈值顺序或权和非法抛 `ValueError`。对 `μ=H[k][i]`、`σ=W[k][i]`：`σ>0` 时 `qj=erfc((thresholds[j]−μ)/(σ*sqrt(2)))/2`，否则 `thresholds[j]≤μ` 取 `1.0`、否则 `0.0`；`p[k][i]=[1−q0, q0−q1, q1−q2, q2]`；`d[k][l]` 自 `[1]+[0]*N` 起按 `d′[r]=d[r](1−p[k][i][l])+(r>0?d[r−1]*p[k][i][l]:0)` 逐受体折入，得 `d[k][l][r]`。`A[k][l][r]=w[k]*d[k][l][r]`、`P[l][r]=fsum(A[k][l][r] for k in 0..K−1)`、`PS[k][l][r]=A[k][l][r]/P[l][r]`（`P[l][r]==0` 时取 `0.0`）；`B[l]=fsum(r*P[l][r] for r in 0..N)`、`ES[k][l][r]=A[k][l][r]*r/B[l]`（`B[l]==0` 时取 `0.0`）。返回情景×等级×计数序的 `(PS, ES)`，均为 K×4×(N+1) `list[list[list[float]]]`，不舍入；`P[l][r]>0` 时各 `(l,r)` 列 `PS` 跨情景求和为 1，`B[l]>0` 时各等级 `ES` 跨情景与计数求和为 1
  - `receptor_excess_interval(H, W, thresholds, weights=None, z=1.96)`：分受体、分阈值的期望超额的跨情景加权均值与区间。`H`、`W` 为同形 K×N 矩阵（`H` 元素可负，`W` 元素非负）；`thresholds` 为 3 个严格递增非负数；`weights` 为 `None`（取等权 `1/K`）或按 `fsum` 归一的 K 个非负权；`z` 为非负区间倍数。对 `μ=H[k][i]`、`σ=W[k][i]`、`t=thresholds[j]`：`σ>0` 时 `a=(t−μ)/σ`、`φ=exp(−a²/2)/sqrt(2π)`、`p=erfc(a/sqrt(2))/2`、`e=σφ+(μ−t)p`、`s=((μ−t)²+σ²)p+σ(μ−t)φ−e²`；`σ=0` 时 `e=max(μ−t,0)`、`s=0`。`M[i][j]=fsum(w[k]*e[k][i][j])`、`V[i][j]=fsum(w[k]*((e[k][i][j]−M[i][j])²+s[k][i][j]))`、`R=z*sqrt(V)`、`L=M−R`、`U=M+R`，返回 N×3 `list[list[float]]` 的 `(M, R, L, U)`，均不舍入
  - `receptor_excess_warning(H, W, thresholds, weights=None, z=1.96)`：由分受体期望超额区间导出的预警等级。五参与 `receptor_excess_interval` 同契约（各类 `TypeError`/`ValueError` 逐项沿用）。令 `(M,R,L,U)=receptor_excess_interval(H,W,thresholds,weights,z)`，则 `levels[i]=Σ[L[i][j]>0]`（`j<3`），`triggers[i]` 在 `levels[i]>0` 时为 `thresholds[levels[i]-1]`、否则为 `None`，`scores=M`，`interval=(L,U)`。返回 `(levels, triggers, scores, interval)`：`levels` 为 N 长 `list[int]`，`triggers` 为 N 长 `list[float|None]`，`scores` 为 N×3 `list[list[float]]`，`interval` 为二元 tuple 且两项均 N×3 `list[list[float]]`，按受体/阈值序，不舍入
  - `receptor_mitigation_plan(H, W, thresholds, options, budget, weights=None, z=1.96)`：在预算内为每个受体选择一个缓解方案。`options` 为长 N 的 list/tuple，`options[i]` 为非空 list/tuple，每项为二元 `(r, c)`（`r`、`c` 均为有限非负数，分别为受体 `i` 的削减量与成本）；`budget` 为有限非负数。外层/分组非序列抛 `TypeError`，长度不符、空分组、项非二元抛 `ValueError`，`r`、`c`、`budget` 类型错抛 `TypeError`，非有限或负抛 `ValueError`；五参与 `receptor_excess_interval` 同契约。对每种每受体选一方案的组合，`total_cost=fsum(c_i)`，超预算舍弃；令 `H′[k][i]=H[k][i]−r_i`，调用 `receptor_excess_interval(H′, W, thresholds, weights, z)` 得 `(M,R,L,U)`，`score=fsum(max(U[i][j],0))`、`trigger=Σ[L[i][j]>0]`（`i<N, j<3`），按 `(score, trigger, total_cost, choice)` 字典序取最小；无可行组合抛 `ValueError`。返回 `(choice, total_cost, score, interval)`：`choice` 为 N 长 0 基选项下标的 `list[int]`，`total_cost`、`score` 为 float，`interval=(M,R,L,U)`，均不舍入
  - `receptor_mitigation_frontier(H, W, thresholds, options, budget, weights=None, z=1.96)`：枚举预算内每受体选一方案的非支配缓解方案前沿。`options`/`budget` 契约及 `TypeError`/`ValueError` 与 `receptor_mitigation_plan` 相同（外层/分组非序列抛 `TypeError`，长度不符、空分组、项非二元抛 `ValueError`，`r`、`c`、`budget` 类型错抛 `TypeError`，非有限或负抛 `ValueError`）；五参与 `receptor_excess_interval` 同契约。对每种组合，`C=fsum(c_i)≤budget` 时令 `H′[k][i]=H[k][i]−r_i`，调用 `receptor_excess_interval(H′, W, thresholds, weights, z)` 得 `(M,R,L,U)`，`S=fsum(max(U[i][j],0))`、`D=fsum(R[i][j])`、`G=Σ[L[i][j]>0]`（`i<N, j<3`）。仅保留在 `(S,D,G,C)` 上逐项 `≤` 且至少一项严格更小的方案不被任何其他方案支配者；无可行组合抛 `ValueError`。返回按 `(S,D,G,C,choice)` 升序排列的 list，每项为 `(choice, C, S, D, G, (M,R,L,U))`，`choice` 为 N 长 0 基选项下标的 `list[int]`，均不舍入
  - `quantile(H, W, quantiles, weights=None, z=1.96)`：情景总健康影响的加权分位数及区间
  - `receptor_quantile(H, W, quantiles, weights=None, z=1.96)`：分受体的加权分位数及区间
  - `receptor_risk_share(H, W, thresholds, weights=None)`：分情景、分受体的超标概率与期望超额占比，返回 K×N×3 的 `(PS, ES)`
  - `receptor_risk_quantile(H, W, thresholds, quantiles, weights=None)`：分受体、分阈值的超标概率 `v(k,i,j)` 的跨情景加权分位数；对每个 `(i,j)` 将各情景按 `(v,k)` 升序排列，`r=0` 取首项，否则取累计归一权首次 `>=r` 的情景，返回分位×受体×阈值序的 M×N×3 `list[list[list[float]]]`，不舍入
  - `sensitivity(H, W, weights=None, z=1.96)`：各受体加权均值、区间半宽及各情景的均值/方差敏感性贡献。`H`、`W` 为同形 K×N 矩阵（`H` 元素可负，`W` 元素非负）；`weights` 为 `None`（取等权 `1/K`）或按 `fsum` 归一的 K 个非负权；`z` 为非负区间倍数。返回 `(M, R, C, S)`，其中
    `M[i] = fsum(w[k]*H[k][i])`、
    `R[i] = z*sqrt(V[i])`、
    `C[k][i] = w[k]*(H[k][i]-M[i])`、
    `V[i] = fsum(w[k]*((H[k][i]-M[i])**2 + W[k][i]**2))`，
    `S[k][i]` 在 `V[i]==0` 时为 `0.0`，否则为 `w[k]*W[k][i]**2/V[i]`；
    `M`、`R` 为 N 长 `list[float]`，`C`、`S` 为 K×N `list[list[float]]`，均不舍入。
