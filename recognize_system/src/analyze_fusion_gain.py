# -*- coding: utf-8 -*-
"""
诊断：为什么"融合特征"相对"纯深度特征"没有提升？

论文 2.3 节的结论是：融合几何特征后（128 → 138 维）分类效果进一步提升。
本地复现中，融合与纯深度的 AUC 在 4 个模型上几乎逐位相同（见检验 B）。
本脚本用三个独立角度检验这个现象。

检验 A —— 信息冗余度
    用 128 维深度特征去回归每个几何特征（岭回归，alpha 由内层 CV 选择）。
    若 R² 明显高于"打乱配对"基线，说明几何特征携带的信息
    已部分包含在深度特征中，拼接带来的新信息有限。

    ⚠️ alpha 必须让 CV 自己选。经验：128 维回归 117 样本时，
       alpha 太小（如 1.0）会严重过拟合，R² 会假性跌到负值，
       从而把"高度冗余"误判成"无冗余"。

检验 B —— 逐折配对比较
    读 10 折的逐折 AUC，看融合相对深度到底有多少折变好、平均差多少。

检验 C —— 错误重叠（最直观）
    取两个分支的 out-of-fold 预测，看它们分错的样本是否高度重叠。
    若几何特征和深度特征用的是同一份信息，两边的错误应该大量重合，
    且"几何能救回深度错判"的样本数不会多 —— 这正是融合无增益的直接原因。
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SRC = DATA / "features_138.csv"
RESULTS = ROOT / "results" / "fusion138"

# 中文名（用于控制台输出）→ 英文 slug（用于文件名）
MODEL_SLUG = {
    "随机森林": "RandomForest",
    "支持向量机": "SVM",
    "XGBoost": "XGBoost",
    "多层感知器": "MLP",
}

SEED = 42
N_SPLITS = 5
# 岭回归 alpha 候选（对数网格）。跨度要足够大，让内层 CV 自己挑。
ALPHAS = np.logspace(0, 4, 17)

# 检验 C 用的模型（MLP 是论文里的最佳模型，SVM 是本地最佳）
PROBE_MODELS = {
    "支持向量机": lambda: SVC(kernel="rbf", C=1.0, gamma="scale",
                              probability=True, random_state=SEED),
    "多层感知器": lambda: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000,
                                    random_state=SEED, early_stopping=False),
}

META = {"image_id", "subject_id", "label", "raw_path", "processed_path"}


def is_deep_col(name):
    return name.startswith("e") and name[1:].isdigit()


def cv_r2(X, y, seed=SEED):
    """交叉验证 R²。标准化 + 岭回归（alpha 由内层 CV 在 ALPHAS 中选）。"""
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    return float(cross_val_score(model, X, y, cv=kf, scoring="r2").mean())


def test_a_redundancy(df, geo_cols, deep_cols):
    print("=" * 74)
    print("检验 A：10 个几何特征能否由 128 维深度特征预测出来？")
    print("=" * 74)
    print(f"岭回归 128 → 1（alpha 内层 CV 选择），{N_SPLITS} 折交叉验证 R²")
    print("对照基线：打乱样本配对（破坏对应关系），R² 应≈0\n")

    X_deep = df[deep_cols].to_numpy(dtype=float)
    rng = np.random.default_rng(SEED)

    rows = []
    for name in geo_cols:
        y = df[name].to_numpy(dtype=float)
        r2_real = cv_r2(X_deep, y)
        # 多次打乱取平均，避免单次打乱残留偶然相关
        r2_shuf = float(np.mean([cv_r2(X_deep, rng.permutation(y)) for _ in range(5)]))
        rows.append({"几何特征": name, "真实配对R²": r2_real,
                     "打乱基线R²": r2_shuf, "冗余度": r2_real - r2_shuf})

    res = pd.DataFrame(rows).sort_values("真实配对R²", ascending=False)
    print(res.to_string(index=False, float_format=lambda v: f"{v:7.3f}"))
    print()
    m = res["真实配对R²"].mean()
    print(f"平均真实配对 R² : {m:+.3f}")
    print(f"平均打乱基线 R² : {res['打乱基线R²'].mean():+.3f}")
    print(f"→ 深度特征平均可解释几何特征方差的 {m * 100:.1f}%")
    print()
    if m > 0.15:
        print("  ✅ 存在明显冗余：约三到五成几何特征方差可由深度特征还原")
    else:
        print("  ❌ 冗余度低：几何特征的信息基本不在深度特征里")

    res.to_csv(RESULTS / "redundancy_geo_from_deep.csv",
               index=False, encoding="utf-8-sig")
    return res


def test_b_paired_folds():
    print()
    print("=" * 74)
    print("检验 B：逐折配对比较 —— 融合 vs 纯深度")
    print("=" * 74)

    rows = []
    for m in ["随机森林", "支持向量机", "XGBoost", "多层感知器"]:
        f_deep = RESULTS / f"perfold_deep_{MODEL_SLUG[m]}.csv"
        f_fuse = RESULTS / f"perfold_fusion_{MODEL_SLUG[m]}.csv"
        if not (f_deep.exists() and f_fuse.exists()):
            print(f"⚠️ 缺少 {m} 的逐折文件，跳过")
            continue
        a = pd.read_csv(f_deep, encoding="utf-8-sig")["AUC"].to_numpy()
        b = pd.read_csv(f_fuse, encoding="utf-8-sig")["AUC"].to_numpy()
        d = b - a
        rows.append({
            "建模方法": m, "深度(128)": a.mean(), "融合(138)": b.mean(),
            "平均提升": d.mean(), "提升折数": int((d > 1e-9).sum()),
            "下降折数": int((d < -1e-9).sum()), "持平折数": int((np.abs(d) <= 1e-9).sum()),
            "逐折AUC标准差": a.std(),
        })

    res = pd.DataFrame(rows)
    print(res.to_string(index=False, float_format=lambda v: f"{v:7.3f}"))
    print()
    print("（共 10 折。若融合真有增益，应「提升折数」远多于「下降折数」，")
    print("  且「平均提升」应显著大于「逐折AUC标准差」；实测相反。）")
    res.to_csv(RESULTS / "paired_folds_deep_vs_fusion.csv",
               index=False, encoding="utf-8-sig")
    return res


def oof_predictions(make_model, X, y, fold_ids):
    """按同一套折划分，返回 out-of-fold 预测（每折只在训练折上拟合标准化器）。"""
    oof = np.full(len(y), -1, dtype=int)
    for f in np.unique(fold_ids):
        te = fold_ids == f
        tr = ~te
        scaler = StandardScaler().fit(X[tr])
        model = make_model()
        model.fit(scaler.transform(X[tr]), y[tr])
        oof[te] = model.predict(scaler.transform(X[te]))
    return oof


def mcnemar(b, c):
    """McNemar 检验（连续性校正）。b/c = 一方错另一方对的样本数。"""
    if b + c == 0:
        return 0.0, 1.0
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    # 卡方(1) 的右尾 p 值
    from math import erfc, sqrt
    p = erfc(sqrt(stat / 2.0))
    return float(stat), float(p)


def test_c_error_overlap(df, geo_cols, deep_cols, fold_ids):
    print()
    print("=" * 74)
    print("检验 C：两个分支分错的样本是否高度重叠？")
    print("=" * 74)

    y = df["label"].map({"digeorge": 0, "down": 1, "williams": 2}).to_numpy()
    X_geo = df[geo_cols].to_numpy(dtype=float)
    X_deep = df[deep_cols].to_numpy(dtype=float)
    X_fuse = np.hstack([X_geo, X_deep])

    rows = []
    for name, factory in PROBE_MODELS.items():
        p_geo = oof_predictions(factory, X_geo, y, fold_ids)
        p_deep = oof_predictions(factory, X_deep, y, fold_ids)
        p_fuse = oof_predictions(factory, X_fuse, y, fold_ids)

        e_geo, e_deep, e_fuse = p_geo != y, p_deep != y, p_fuse != y
        both_err = int((e_geo & e_deep).sum())
        # 几何与深度错误的 Jaccard 重叠系数
        union = int((e_geo | e_deep).sum())
        jac = both_err / union if union else 1.0

        # 融合 vs 深度：McNemar
        fix = int((e_deep & ~e_fuse).sum())     # 融合改正的
        brk = int((~e_deep & e_fuse).sum())     # 融合新引入的错误
        stat, p = mcnemar(fix, brk)

        rows.append({
            "建模方法": name,
            "几何错": int(e_geo.sum()),
            "深度错": int(e_deep.sum()),
            "两者都错": both_err,
            "错误重叠率": jac,
            "融合改正": fix,
            "融合致错": brk,
            "净收益": fix - brk,
            "McNemar p": p,
        })

    res = pd.DataFrame(rows)
    print(res.to_string(index=False, float_format=lambda v: f"{v:7.3f}"))
    print()
    print("「错误重叠率」= 两者都错 / 至少一方错。若两分支用的是同一份信息，")
    print("该值应接近 1；「融合改正」应远多于「融合致错」且 p < 0.05。")
    res.to_csv(RESULTS / "error_overlap.csv", index=False, encoding="utf-8-sig")
    return res


def main():
    df = pd.read_csv(SRC, encoding="utf-8-sig")
    folds = pd.read_csv(DATA / "group_folds_10.csv", encoding="utf-8-sig")
    df = df.merge(folds[["image_id", "fold"]], on="image_id", how="inner")

    deep_cols = [c for c in df.columns if is_deep_col(c)]
    geo_cols = [c for c in df.columns if c not in deep_cols and c not in META | {"fold"}]
    print(f"输入: {SRC.name}   几何 {len(geo_cols)} 维 + 深度 {len(deep_cols)} 维"
          f"   样本 {len(df)}\n")

    test_a_redundancy(df, geo_cols, deep_cols)
    test_b_paired_folds()
    test_c_error_overlap(df, geo_cols, deep_cols, df["fold"].to_numpy())


if __name__ == "__main__":
    main()
