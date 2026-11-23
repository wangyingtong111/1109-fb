# -*- coding: utf-8 -*-
"""
FeatureUnion + imblearn.Pipeline(RandomUnderSampler -> StandardScaler)
行身份（row identity）取证脚本（离线、确定性、可反复执行）

要回答的问题：
    两路特征各自独立下采样后再 hstack，合并结果里同一行的两列是否还来自
    同一条原始记录？问题出在 采样 / 拟合 / 拼接 的哪一环？

实验设计
--------
E1  fit_transform 路径，两路 RUS 使用【不同】固定种子
E2  同一结构改成 fit() 之后 transform()（训练数据再走一遍）
E3  E2 拟合好的管道，对两条【新记录】做 transform
E4  对照：两路 RUS 使用【同一】索引（同种子），仍走 fit_transform

每条样本都带唯一行标记 row_id（第 0 列），两路都保留它。StandardScaler
是可逆仿射变换，用 scaler.inverse_transform 即可从缩放后的输出中无损还原
row_id，从而追踪每一路输出的真实行序。

只依赖 numpy / scikit-learn / imbalanced-learn，不依赖 pandas。
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler
import imblearn

# ----------------------------------------------------------------------
# 0. 固定的取证数据：7 条带唯一行标记的样本，标签 5 个 0 + 2 个 1
# ----------------------------------------------------------------------
ROW_ID = np.arange(7)
Y = np.array([0, 0, 0, 0, 0, 1, 1])

# 每一路都把 row_id 放在第 0 列；其余列取便于人工核对的可识别数值。
# 模拟真实场景：两路特征来自同一张 7 行宽表，row_id 是共享的行身份列。
XA = np.column_stack([ROW_ID,
                      [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
                      [11.0, 21.0, 31.0, 41.0, 51.0, 61.0, 71.0]])
XB = np.column_stack([ROW_ID,
                      [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0],
                      [101.0, 201.0, 301.0, 401.0, 501.0, 601.0, 701.0]])
# 合并宽表（5 列）：列 0=共享 row_id，列 1-2=A 路特征，列 3-4=B 路特征
X = np.column_stack([XA, XB[:, 1:]])


class ColumnSelector(BaseEstimator, TransformerMixin):
    """按列下标取列；放在管道首位，使两路能各自选取自己的特征块。"""

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.columns]


def make_branch(seed):
    """复现同事的结构：... -> RandomUnderSampler -> StandardScaler。"""
    return make_pipeline(
        RandomUnderSampler(random_state=seed),
        StandardScaler(),
    )


def build_union(seed_a, seed_b):
    return FeatureUnion([
        ("branch_A", make_pipeline(ColumnSelector([0, 1, 2]), make_branch(seed_a))),
        ("branch_B", make_pipeline(ColumnSelector([0, 3, 4]), make_branch(seed_b))),
    ])


def inner_pipe(union, name):
    """取出 FeatureUnion 某一路内部的 imblearn Pipeline（兼容新旧 sklearn）。"""
    transformers = getattr(union, "named_transformers_",
                           getattr(union, "named_transformers"))
    return transformers[name][1]


def scaler_of(branch_pipe):
    """取出嵌套在 imblearn Pipeline 末尾的 StandardScaler。"""
    return branch_pipe.named_steps["standardscaler"]


def rus_of(branch_pipe):
    return branch_pipe.named_steps["randomundersampler"]


def recover_ids(branch_pipe, X_out):
    """逆标准化，取第 0 列并四舍五入，还原输出每一行的原始 row_id。"""
    raw = scaler_of(branch_pipe).inverse_transform(X_out)
    return np.rint(raw[:, 0]).astype(int)


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def show_branch(name, branch_pipe, X_out):
    idx = rus_of(branch_pipe).sample_indices_
    means = scaler_of(branch_pipe).mean_
    ids = recover_ids(branch_pipe, X_out)
    print(f"[{name}] RUS.sample_indices_ (既是选中索引, 也是输出行序): {idx.tolist()}")
    print(f"[{name}] StandardScaler.mean_ (含 row_id 列)           : {np.round(means, 4).tolist()}")
    print(f"[{name}] 输出形状={X_out.shape}  逆变换还原的真实行序 row_id : {ids.tolist()}")
    return idx, means, ids


def merge_report(ids_a, ids_b):
    """逐位置比对两路 row_id，列出合并后错配的行。"""
    rows = []
    n_mismatch = 0
    for pos, (ia, ib) in enumerate(zip(ids_a, ids_b)):
        ok = (ia == ib)
        n_mismatch += (not ok)
        rows.append((pos, ia, ib, "OK" if ok else "MISMATCH"))
    print("\n合并后逐位置身份比对（FeatureUnion 只按位置 hstack，不做索引对齐）:")
    print(f"  {'输出位置':>6} | {'A路row_id':>8} | {'B路row_id':>8} | 结论")
    for pos, ia, ib, verdict in rows:
        print(f"  {pos:>6} | {ia:>8} | {ib:>8} | {verdict}")
    print(f"  共 {len(rows)} 行，错配 {n_mismatch} 行；两路行数均为 {len(ids_a)}。")
    return n_mismatch


# 运行时找一对“确实抽到不同索引”的种子，保证 E1 的错配真实出现且与版本无关
def find_distinct_seeds():
    base = RandomUnderSampler(random_state=0)
    base.fit_resample(XA, Y)
    idx0 = base.sample_indices_.copy()
    for s in range(1, 1000):
        rus = RandomUnderSampler(random_state=s)
        rus.fit_resample(XA, Y)
        if not np.array_equal(rus.sample_indices_, idx0):
            return 0, s, idx0, rus.sample_indices_.copy()
    raise RuntimeError("未找到产生不同索引的种子对")


print("版本: imblearn", imblearn.__version__,
      "| sklearn", __import__("sklearn").__version__,
      "| numpy", np.__version__)
print("原始数据 XA:\n", XA)
print("原始数据 XB:\n", XB)
print("送入 FeatureUnion 的宽表 X（列0=共享row_id, 列1-2=A路, 列3-4=B路）:\n", X)
print("标签 y:", Y.tolist(), "（多数类 0 有 5 条，少数类 1 有 2 条）")

SEED_A, SEED_B, IDX_A_PREVIEW, IDX_B_PREVIEW = find_distinct_seeds()
print(f"\n选定种子: A路={SEED_A}, B路={SEED_B}")
print(f"预检索引 A={IDX_A_PREVIEW.tolist()}  B={IDX_B_PREVIEW.tolist()} "
      f"-> 不同: {not np.array_equal(IDX_A_PREVIEW, IDX_B_PREVIEW)}")

# ----------------------------------------------------------------------
# E1  fit_transform 路径：两路不同种子（复现事故现场）
# ----------------------------------------------------------------------
section("E1  FeatureUnion.fit_transform，两路 RUS 不同固定种子（事故现场）")
union1 = build_union(SEED_A, SEED_B)
Z1 = union1.fit_transform(X, Y)              # 注意 y 被送进每一路管道
za1, zb1 = np.hsplit(Z1, [3])

idx_a1, mean_a1, ids_a1 = show_branch("A", inner_pipe(union1, "branch_A"), za1)
idx_b1, mean_b1, ids_b1 = show_branch("B", inner_pipe(union1, "branch_B"), zb1)
n_bad1 = merge_report(ids_a1, ids_b1)

# 合并后的标签位置（按类块顺序：先类0后类1，两路完全一致）
print("\n位置标签核对: 两路输出标签序列均为",
      Y[rus_of(inner_pipe(union1, "branch_A")).sample_indices_].tolist(),
      "-> 只看类别标签看不出错配，必须看 row_id。")

assert za1.shape[0] == zb1.shape[0] == 4, "下采样后每路都应是 4 行"
assert n_bad1 > 0, "不同种子应当产生至少一处行身份错配"
assert not np.array_equal(idx_a1, idx_b1), "两路采样索引必须不同"

# ----------------------------------------------------------------------
# E2  同一结构改成 fit 后 transform（训练数据再走一遍）
# ----------------------------------------------------------------------
section("E2  同一结构改为 fit() 后 transform()（对同一批训练数据）")
union2 = build_union(SEED_A, SEED_B)
union2.fit(X, Y)                             # fit 阶段采样仍发生，scaler 在采样子集上拟合
Z2 = union2.transform(X)                     # transform 阶段：采样器被 _iter() 过滤，不再采样
za2, zb2 = np.hsplit(Z2, [3])

pipe_a2 = inner_pipe(union2, "branch_A")
pipe_b2 = inner_pipe(union2, "branch_B")

# fit 阶段记录到的采样索引与 scaler 均值（与 E1 同种子时应一致）
idx_a2 = rus_of(pipe_a2).sample_indices_
idx_b2 = rus_of(pipe_b2).sample_indices_
mean_a2 = scaler_of(pipe_a2).mean_
mean_b2 = scaler_of(pipe_b2).mean_
print(f"[A] fit 阶段采样索引: {idx_a2.tolist()}  scaler 均值: {np.round(mean_a2, 4).tolist()}")
print(f"[B] fit 阶段采样索引: {idx_b2.tolist()}  scaler 均值: {np.round(mean_b2, 4).tolist()}")

ids_a2 = recover_ids(pipe_a2, za2)
ids_b2 = recover_ids(pipe_b2, zb2)
print(f"[A] transform 输出形状={za2.shape} 还原行序: {ids_a2.tolist()}")
print(f"[B] transform 输出形状={zb2.shape} 还原行序: {ids_b2.tolist()}")
n_bad2 = merge_report(ids_a2, ids_b2)

assert za2.shape[0] == zb2.shape[0] == 7, "transform 不采样，行数应回到原始 7 行"
assert n_bad2 == 0, "transform 路径两路都按原始输入行序，必须完全配对"
assert np.array_equal(ids_a2, ROW_ID) and np.array_equal(ids_b2, ROW_ID)
# 同种子下，fit 阶段的采样结果与 scaler 拟合均值必须和 E1 完全一致（确定性）
assert np.array_equal(idx_a2, idx_a1) and np.array_equal(idx_b2, idx_b1)
assert np.allclose(mean_a2, mean_a1) and np.allclose(mean_b2, mean_b1)

# ----------------------------------------------------------------------
# E3  对两条新记录做 transform：验证线上推理是否仍按输入行配对
# ----------------------------------------------------------------------
section("E3  用 E2 拟合好的管道 transform 两条【新记录】")
XNA = np.array([[7.0, 80.0, 81.0],
                [8.0, 90.0, 91.0]])
XNB = np.array([[7.0, 800.0, 801.0],
                [8.0, 900.0, 901.0]])
XN = np.column_stack([XNA, XNB[:, 1:]])     # 与训练宽表同构的两条新记录
ZN = union2.transform(XN)
zna, znb = np.hsplit(ZN, [3])
ids_na = recover_ids(pipe_a2, zna)
ids_nb = recover_ids(pipe_b2, znb)
print(f"新记录 A 路还原 row_id: {ids_na.tolist()}")
print(f"新记录 B 路还原 row_id: {ids_nb.tolist()}")
print("输入顺序即输出顺序，两路按输入位置一一配对，",
      "无采样、无重排。" if np.array_equal(ids_na, ids_nb) else "出现配对错误！")
assert np.array_equal(ids_na, [7, 8]) and np.array_equal(ids_nb, [7, 8])
# scaler 未被新数据改动
assert np.allclose(scaler_of(pipe_a2).mean_, mean_a2)
assert np.allclose(scaler_of(pipe_b2).mean_, mean_b2)

# ----------------------------------------------------------------------
# E4  对照：两路使用同一索引（同种子），仍走 fit_transform
# ----------------------------------------------------------------------
section("E4  对照实验：两路 RUS 同种子 -> 同一采样索引（fit_transform）")
union4 = build_union(SEED_A, SEED_A)
Z4 = union4.fit_transform(X, Y)
za4, zb4 = np.hsplit(Z4, [3])
pa4 = inner_pipe(union4, "branch_A")
pb4 = inner_pipe(union4, "branch_B")
idx_a4, mean_a4, ids_a4 = show_branch("A", pa4, za4)
idx_b4, mean_b4, ids_b4 = show_branch("B", pb4, zb4)
n_bad4 = merge_report(ids_a4, ids_b4)

assert np.array_equal(idx_a4, idx_b4), "同种子必须抽到完全相同的索引与行序"
assert n_bad4 == 0, "索引一致时 hstack 不会错配"

# ----------------------------------------------------------------------
# 汇总：为什么只看行数会漏掉问题
# ----------------------------------------------------------------------
section("汇总：行数检查 vs 身份检查")
print(f"E1 不同种子 fit_transform : 两路行数 {za1.shape[0]} == {zb1.shape[0]} "
      f"-> 行数检查通过；身份错配 {n_bad1} 行")
print(f"E2 同结构 fit+transform   : 两路行数 {za2.shape[0]} == {zb2.shape[0]} "
      f"-> 行数检查通过；身份错配 {n_bad2} 行")
print(f"E3 新记录 transform       : 两路行数 {zna.shape[0]} == {znb.shape[0]} "
      f"-> 行数检查通过；身份错配 0 行")
print(f"E4 同种子 fit_transform   : 两路行数 {za4.shape[0]} == {zb4.shape[0]} "
      f"-> 行数检查通过；身份错配 {n_bad4} 行")
print("\n结论: RandomUnderSampler 永远把每类平衡到相同数量，所以无论种子是否相同、")
print("无论 fit_transform 还是 transform，两路行数都相等。行数只能证明‘位置数’，")
print("无法证明同一位置上的两列来自同一条原始记录——必须比对 row_id 这类身份标记。")

print("\n全部断言通过。")
