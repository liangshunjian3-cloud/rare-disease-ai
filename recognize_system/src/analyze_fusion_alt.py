# -*- coding: utf-8 -*-
"""
诊断（续）：如果两个分支的错误是互补的，为什么直接拼接没有增益？

analyze_fusion_gain.py 的检验 C 显示：几何分支与深度分支分错的样本
重叠率只有 0.17~0.28，说明两者用的是【不同】的信息。但简单拼接
（X = [geo ; deep]）并没有把这份互补性转成提升。

本脚本验证两种替代融合方式：

检验 D —— 加权拼接
    拼接后几何块只有 10 维、深度块 128 维，几何块可能被淹没。
    把几何块整体乘以权重 w 后再拼接，看是否能拉开差距。

检验 E —— 概率集成（对这个数据结构更合理的融合方式）
    p = w · p_几何 + (1-w) · p_深度
    两个模型各自独立训练，只在【概率输出】层面融合。
    这是利用"互补错误"的标准做法。

对照基准：纯深度特征（论文里融合要超越的就是它）。
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SRC = DATA / "features_138.csv"
RESULTS = ROOT / "results" / "fusion138"

SEED = 42
WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]

MODEL_FACTORY = {
    "随机森林": lambda: RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
    "支持向量机": lambda: SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=SEED),
    "多层感知器": lambda: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000,
                                    random_state=SEED, early_stopping=False),
}
LABEL_MAP = {"digeorge": 0, "down": 1, "williams": 2}
META = {"image_id", "subject_id", "label", "raw_path", "processed_path"}


def is_deep_col(name):
    return name.startswith("e") and name[1:].isdigit()


def micro_auc(y_true, y_prob):
    y_bin = np.zeros((len(y_true), y_prob.shape[1]))
    y_bin[np.arange(len(y_true)), y_true] = 1
    fpr, tpr, _ = roc_curve(y_bin.ravel(), y_prob.ravel())
    return float(auc(fpr, tpr))


def oof_proba(make_model, X, y, fold_ids):
    """out-of-fold 概率（每折只在训练折上拟合标准化器）。"""
    oof = np.zeros((len(y), 3))
    for f in np.unique(fold_ids):
        te, tr = fold_ids == f, fold_ids != f
        sc = StandardScaler().fit(X[tr])
        m = make_model()
        m.fit(sc.transform(X[tr]), y[tr])
        assert list(m.classes_) == [0, 1, 2], f"折 {f} 类别不全: {m.classes_}"
        oof[te] = m.predict_proba(sc.transform(X[te]))
    return oof


def test_d_weighted_concat(X_geo, X_deep, y, fold_ids):
    print("=" * 74)
    print("检验 D：加权拼接 —— 放大几何块能否改善？")
    print("=" * 74)
    print("几何块 × w 后与深度块拼接（标准化在拼接后统一做，故各列等方差）\n")

    rows = []
    for name, factory in MODEL_FACTORY.items():
        base = micro_auc(y, oof_proba(factory, X_deep, y, fold_ids))
        row = {"建模方法": name, "纯深度": base}
        for w in WEIGHTS:
            X = np.hstack([X_geo * w, X_deep])
            row[f"w={w:g}"] = micro_auc(y, oof_proba(factory, X, y, fold_ids))
        row["最佳w"] = max(WEIGHTS, key=lambda w: row[f"w={w:g}"])
        row["最佳提升"] = row[f"w={row['最佳w']:g}"] - base
        rows.append(row)

    res = pd.DataFrame(rows)
    show = ["建模方法", "纯深度"] + [f"w={w:g}" for w in WEIGHTS] + ["最佳w", "最佳提升"]
    print(res[show].to_string(index=False, float_format=lambda v: f"{v:7.3f}"))
    print()
    res.to_csv(RESULTS / "weighted_concat.csv", index=False, encoding="utf-8-sig")
    return res


def test_e_prob_ensemble(X_geo, X_deep, y, fold_ids):
    print()
    print("=" * 74)
    print("检验 E：概率集成 —— p = w·p_几何 + (1-w)·p_深度")
    print("=" * 74)
    print("w=0 等价于纯深度，w=1 等价于纯几何。两个分支各自独立训练。\n")

    rows = []
    for name, factory in MODEL_FACTORY.items():
        p_geo = oof_proba(factory, X_geo, y, fold_ids)
        p_deep = oof_proba(factory, X_deep, y, fold_ids)
        row = {"建模方法": name, "几何AUC": micro_auc(y, p_geo),
               "深度AUC": micro_auc(y, p_deep)}
        for w in WEIGHTS:
            row[f"w={w:g}"] = micro_auc(y, w * p_geo + (1 - w) * p_deep)
        row["最佳w"] = max(WEIGHTS, key=lambda w: row[f"w={w:g}"])
        row["最佳AUC"] = row[f"w={row['最佳w']:g}"]
        row["较深度提升"] = row["最佳AUC"] - row["深度AUC"]
        rows.append(row)

    res = pd.DataFrame(rows)
    show = (["建模方法", "几何AUC", "深度AUC"]
            + [f"w={w:g}" for w in WEIGHTS] + ["最佳w", "最佳AUC", "较深度提升"])
    print(res[show].to_string(index=False, float_format=lambda v: f"{v:7.3f}"))
    print()
    print("（注意：最佳 w 是在【同一批 out-of-fold 预测】上挑的，")
    print("  存在选择性偏差，真实泛化增益会小于表中数值。）")
    res.to_csv(RESULTS / "prob_ensemble.csv", index=False, encoding="utf-8-sig")
    return res


def main():
    df = pd.read_csv(SRC, encoding="utf-8-sig")
    folds = pd.read_csv(DATA / "group_folds_10.csv", encoding="utf-8-sig")
    df = df.merge(folds[["image_id", "fold"]], on="image_id", how="inner")

    deep_cols = [c for c in df.columns if is_deep_col(c)]
    geo_cols = [c for c in df.columns if c not in deep_cols and c not in META | {"fold"}]

    y = df["label"].map(LABEL_MAP).to_numpy()
    fold_ids = df["fold"].to_numpy()
    X_geo = df[geo_cols].to_numpy(dtype=float)
    X_deep = df[deep_cols].to_numpy(dtype=float)

    print(f"输入: {SRC.name}   几何 {len(geo_cols)} 维 + 深度 {len(deep_cols)} 维"
          f"   样本 {len(df)}\n")

    test_d_weighted_concat(X_geo, X_deep, y, fold_ids)
    test_e_prob_ensemble(X_geo, X_deep, y, fold_ids)


if __name__ == "__main__":
    main()
