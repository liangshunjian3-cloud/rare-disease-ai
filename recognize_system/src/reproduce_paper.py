# -*- coding: utf-8 -*-
"""
复现《小样本条件下利用面部特征提高罕见病预测准确率的方法研究》

    喻为栋, 徐青云, 刘雷. 中国数字医学, 2024, 19(10): 20-27.
    DOI: 10.3969/j.issn.1673-7571.2024.10.004

--------------------------------------------------------------------------
本脚本覆盖论文的【步骤 7-8】：模型训练 + 评估
    4 种模型（随机森林 / SVM / XGBoost / MLP）
    × 2 种特征方案（全部几何特征 / 折内 RF 选 Top10）
    × 10 折交叉验证（复用 group_folds_10.csv，按 subject 分组）

【未覆盖】论文步骤 1-6（Dlib 关键点 → 61 维几何特征 → 128 维深度特征 → 138 维融合）
    原因：本地无原始图片，且 paper 的 61 维特征定义与现有 30 维不一致。
    详见 inform.md 第八节。

【重要差异】现有 geometric_features.csv 为 30 维自定义几何特征，
    与论文表 1/表 2 的 61 维定义不同。因此本脚本对应的是论文表 4 中
    「仅采用面部几何特征」这一列的近似复现，数值不应与论文逐位对齐。
--------------------------------------------------------------------------
"""

from pathlib import Path
import argparse
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, auc,
)
from xgboost import XGBClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Windows 下 matplotlib 默认字体不含中文，需显式指定，否则图中中文显示为方框
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

warnings.filterwarnings("ignore")

# ==========================================================================
# 配置
# ==========================================================================
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# 特征表 → 结果子目录。短名便于在文件名里使用。
SCHEME_DIR = {
    "geometric_features": "geo30",        # 30 维自定义几何特征
    "geometric_features_61": "geo61",     # 论文表1/表2 的 61 维几何特征
    "features_138": "fusion138",          # 10 几何 + 128 深度
}

# 论文 2.1 节：DiGeorge=0, 唐氏=1, Williams=2
LABEL_MAP = {"digeorge": 0, "down": 1, "williams": 2}
LABEL_NAMES = ["DiGeorge", "Down", "Williams"]
CLASSES = [0, 1, 2]

SEED = 42
N_SELECT = 10  # 论文 2.2 节：随机森林筛选前 10 个最重要特征

# 论文未给出超参，此处取 inform.md 第五节记录的推荐值。
# 用工厂函数而非实例，保证每次实验都是全新的、未拟合的模型。
MODEL_FACTORY = {
    "随机森林": lambda: RandomForestClassifier(
        n_estimators=100, random_state=SEED, n_jobs=-1
    ),
    "支持向量机": lambda: SVC(
        kernel="rbf", C=1.0, gamma="scale",
        probability=True, random_state=SEED,
    ),
    "XGBoost": lambda: XGBClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3,
        random_state=SEED, verbosity=0, n_jobs=-1,
    ),
    "多层感知器": lambda: MLPClassifier(
        hidden_layer_sizes=(64, 32), max_iter=1000,
        random_state=SEED, early_stopping=False,
    ),
}

# 中文名（用于控制台输出与图表标题）→ 英文 slug（用于文件名）
# 文件系统上不用中文，避免 Windows GBK 环境下的编码问题。
MODEL_SLUG = {
    "随机森林": "RandomForest",
    "支持向量机": "SVM",
    "XGBoost": "XGBoost",
    "多层感知器": "MLP",
}
SCHEME_SLUG = {
    "全部特征": "all",
    "Top10特征": "top10",
    "几何特征": "geometric",
    "深度学习特征": "deep",
    "融合特征": "fusion",
}


# ==========================================================================
# 数据加载
# ==========================================================================
def load_data(feature_file="geometric_features.csv"):
    """读入几何特征表与折划分表，返回 (X, y, folds, feature_names)。"""
    feats = pd.read_csv(DATA / feature_file, encoding="utf-8-sig")
    folds = pd.read_csv(DATA / "group_folds_10.csv", encoding="utf-8-sig")

    df = feats.merge(
        folds[["image_id", "fold"]], on="image_id", how="inner"
    )
    if len(df) != len(feats):
        raise RuntimeError(
            f"合并后行数不一致: {len(df)} != {len(feats)}，请检查 image_id"
        )

    meta_cols = {"image_id", "subject_id", "label", "raw_path", "processed_path", "fold"}
    # 排除非论文定义的附加列（extract_features_61.py 输出的对照列）
    exclude = {"bottom_lip_length_alt"}
    feature_names = [c for c in df.columns if c not in meta_cols | exclude]

    X = df[feature_names].to_numpy(dtype=float)
    y = df["label"].map(LABEL_MAP).to_numpy()
    fold_ids = df["fold"].to_numpy()

    if np.isnan(X).any():
        raise RuntimeError("特征矩阵中存在 NaN")
    if np.isnan(y).any():
        raise RuntimeError("存在无法映射的标签")

    return X, y, fold_ids, feature_names


# ==========================================================================
# 单个折的训练与评估
# ==========================================================================
def evaluate_fold(model, X_tr, y_tr, X_te, y_te):
    """在单折上训练并预测，返回 (真实标签, 预测标签, 各类概率)。"""
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)
    return y_te, y_pred, y_prob


def compute_metrics(y_true, y_pred, y_prob):
    """论文 2.3 节指标：准确率 / 精确率 / 召回率 / F1 / AUC(micro)。

    多分类下精确率、召回率、F1 采用 macro 平均；AUC 按论文要求用 micro 平均。
    """
    return {
        "准确率": accuracy_score(y_true, y_pred),
        "精确率": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "召回率": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1值": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "AUC": roc_auc_score(
            y_true, y_prob, multi_class="ovr", average="micro", labels=CLASSES
        ),
    }


# ==========================================================================
# 主流程：10 折交叉验证
# ==========================================================================
def run_cv(model, X, y, fold_ids, use_feature_selection):
    """跑一遍 10 折交叉验证。

    use_feature_selection=True 时，在【每折训练集内部】用随机森林挑 Top10，
    避免在全量数据上做特征选择导致的信息泄漏。

    返回 (逐折指标 DataFrame, 全折真实标签, 全折预测标签, 全折预测概率)。
    """
    all_true, all_pred, all_prob = [], [], []
    fold_rows = []

    for f in sorted(np.unique(fold_ids)):
        tr, te = fold_ids != f, fold_ids == f
        X_tr, y_tr = X[tr], y[tr]
        X_te, y_te = X[te], y[te]

        # 特征选择（仅用训练折，防止泄漏）
        if use_feature_selection:
            selector = RandomForestClassifier(
                n_estimators=100, random_state=SEED, n_jobs=-1
            )
            selector.fit(X_tr, y_tr)
            top_idx = np.argsort(selector.feature_importances_)[::-1][:N_SELECT]
            X_tr, X_te = X_tr[:, top_idx], X_te[:, top_idx]

        # 标准化（同样只在训练折上 fit）
        scaler = StandardScaler().fit(X_tr)
        X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

        y_true, y_pred, y_prob = evaluate_fold(
            model, X_tr_s, y_tr, X_te_s, y_te
        )

        m = compute_metrics(y_true, y_pred, y_prob)
        m["折"] = int(f)
        fold_rows.append(m)

        all_true.append(y_true)
        all_pred.append(y_pred)
        all_prob.append(y_prob)

    return (
        pd.DataFrame(fold_rows),
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.concatenate(all_prob),
    )


def micro_roc(y_true, y_prob):
    """论文 2.3 节：ROC 曲线采用微平均(micro-averaging)方法计算。"""
    y_bin = label_binarize(y_true, classes=CLASSES)
    return roc_curve(y_bin.ravel(), y_prob.ravel())


# ==========================================================================
# 绘图
# ==========================================================================
def plot_roc(curves, feature_tag):
    """绘制 4 模型的 micro-average ROC 曲线（对应论文图 7）。"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, (name, (fpr, tpr, auc_val)) in zip(axes.ravel(), curves.items()):
        ax.plot(fpr, tpr, lw=2, label=f"micro-average ROC (AUC = {auc_val:.2f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{name} 的 ROC 曲线")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle(f"micro-average ROC 曲线（特征方案：{feature_tag}）", fontsize=13)
    fig.tight_layout()
    out = RESULTS_DIR / f"roc_{SCHEME_SLUG.get(feature_tag, feature_tag)}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_confusion(y_true, y_pred, model_name, feature_tag):
    """绘制混淆矩阵。"""
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), LABEL_NAMES)
    ax.set_yticks(range(3), LABEL_NAMES)
    ax.set_xlabel("预测")
    ax.set_ylabel("真实")
    ax.set_title(f"{model_name} 混淆矩阵\n（{feature_tag}，10折汇总）", fontsize=11)
    for i in range(3):
        for j in range(3):
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=13,
            )
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out = RESULTS_DIR / (
        f"confusion_{MODEL_SLUG.get(model_name, model_name)}"
        f"_{SCHEME_SLUG.get(feature_tag, feature_tag)}.png"
    )
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return cm


# ==========================================================================
# 主函数
# ==========================================================================
def main(feature_file="geometric_features.csv"):
    global RESULTS_DIR

    # 每个特征集单独一个输出目录，避免不同特征方案的图表互相覆盖
    RESULTS_DIR = ROOT / "results" / SCHEME_DIR.get(
        Path(feature_file).stem, Path(feature_file).stem
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X, y, fold_ids, feature_names = load_data(feature_file)

    print("=" * 74)
    print("复现：小样本条件下利用面部特征提高罕见病预测准确率的方法研究")
    print("=" * 74)
    print(f"特征表      : {feature_file}")
    print(f"图片数      : {len(y)}")
    print(
        "受试者数    : "
        f"{pd.read_csv(DATA / feature_file, encoding='utf-8-sig')['subject_id'].nunique()}"
    )
    print(f"特征维度    : {X.shape[1]}")
    print(f"类别分布    : {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"折数        : {len(np.unique(fold_ids))}")
    print(f"每折样本数  : {dict(sorted(zip(*np.unique(fold_ids, return_counts=True))))}")
    print()

    summary_rows = []
    for fs_flag, tag in [(False, "全部特征"), (True, "Top10特征")]:
        print("#" * 74)
        print(f"# 特征方案：{tag}"
              f"{f'（折内 RF 选 {N_SELECT} 个）' if fs_flag else f'（{X.shape[1]} 维）'}")
        print("#" * 74)

        curves = {}
        for model_name, make_model in MODEL_FACTORY.items():
            fold_df, y_true, y_pred, y_prob = run_cv(
                make_model(), X, y, fold_ids, fs_flag
            )

            mean = fold_df[["准确率", "精确率", "召回率", "F1值", "AUC"]].mean()
            std = fold_df[["准确率", "精确率", "召回率", "F1值", "AUC"]].std()

            print(f"\n【{model_name}】")
            print(
                f"  精确率 {mean['精确率']:.2f}±{std['精确率']:.2f}   "
                f"召回率 {mean['召回率']:.2f}±{std['召回率']:.2f}   "
                f"F1 {mean['F1值']:.2f}±{std['F1值']:.2f}   "
                f"AUC {mean['AUC']:.2f}±{std['AUC']:.2f}   "
                f"准确率 {mean['准确率']:.2f}"
            )

            fpr, tpr, _ = micro_roc(y_true, y_prob)
            curves[model_name] = (fpr, tpr, auc(fpr, tpr))

            cm = plot_confusion(y_true, y_pred, model_name, tag)
            print(f"  10折汇总混淆矩阵 (行=真实, 列=预测):")
            for i, row in enumerate(cm):
                print(f"    {LABEL_NAMES[i]:>9s}  {row}")

            summary_rows.append({
                "特征方案": tag,
                "建模方法": model_name,
                "精确率": f"{mean['精确率']:.2f}±{std['精确率']:.2f}",
                "召回率": f"{mean['召回率']:.2f}±{std['召回率']:.2f}",
                "F1值": f"{mean['F1值']:.2f}±{std['F1值']:.2f}",
                "AUC平均值": f"{mean['AUC']:.2f}±{std['AUC']:.2f}",
                "准确率": f"{mean['准确率']:.2f}±{std['准确率']:.2f}",
            })

            # 保存逐折明细（文件名用英文 slug）
            fold_df.to_csv(
                RESULTS_DIR
                / f"perfold_{SCHEME_SLUG[tag]}_{MODEL_SLUG[model_name]}.csv",
                index=False, encoding="utf-8-sig",
            )

        roc_path = plot_roc(curves, tag)
        print(f"\n  → ROC 曲线已保存: {roc_path.name}\n")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        RESULTS_DIR / "metrics_summary.csv", index=False, encoding="utf-8-sig"
    )

    print("=" * 74)
    print("结果汇总（对应论文表 4 的「仅几何特征」一列）")
    print("=" * 74)
    print(summary.to_string(index=False))
    print()
    print(f"详细结果已保存至: {RESULTS_DIR}")
    print("=" * 74)

    return summary


def is_deep_col(name: str) -> bool:
    """判断是否为 Dlib 深度特征列（e0, e1, ..., e127）。"""
    return name.startswith("e") and name[1:].isdigit()


def main_fusion(feature_file="features_138.csv"):
    """论文 2.3 节的三种特征方案对照实验。

        几何特征 (10)  /  深度学习特征 (128)  /  融合 (138)
    每种方案 × 4 种模型 × 10 折交叉验证。
    """
    global RESULTS_DIR
    RESULTS_DIR = ROOT / "results" / SCHEME_DIR.get(
        Path(feature_file).stem, Path(feature_file).stem
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X, y, fold_ids, feature_names = load_data(feature_file)
    geo_idx = [i for i, n in enumerate(feature_names) if not is_deep_col(n)]
    deep_idx = [i for i, n in enumerate(feature_names) if is_deep_col(n)]

    print("=" * 74)
    print("复现：论文 2.3 节 —— 三种特征方案对照")
    print("=" * 74)
    print(f"特征表      : {feature_file}")
    print(f"样本数      : {len(y)}")
    print(f"几何特征    : {len(geo_idx)} 维")
    print(f"深度特征    : {len(deep_idx)} 维")
    print(f"融合特征    : {X.shape[1]} 维")
    print(f"类别分布    : {dict(zip(*np.unique(y, return_counts=True)))}")
    print()

    arms = [
        ("几何特征", geo_idx),
        ("深度学习特征", deep_idx),
        ("融合特征", list(range(X.shape[1]))),
    ]

    summary_rows = []
    for arm_name, idx in arms:
        print("#" * 74)
        print(f"# 特征方案：{arm_name}（{len(idx)} 维）")
        print("#" * 74)

        Xa = X[:, idx]
        curves = {}
        for model_name, make_model in MODEL_FACTORY.items():
            fold_df, y_true, y_pred, y_prob = run_cv(
                make_model(), Xa, y, fold_ids, use_feature_selection=False
            )

            mean = fold_df[["准确率", "精确率", "召回率", "F1值", "AUC"]].mean()
            std = fold_df[["准确率", "精确率", "召回率", "F1值", "AUC"]].std()

            print(f"\n【{model_name}】")
            print(
                f"  精确率 {mean['精确率']:.2f}±{std['精确率']:.2f}   "
                f"召回率 {mean['召回率']:.2f}±{std['召回率']:.2f}   "
                f"F1 {mean['F1值']:.2f}±{std['F1值']:.2f}   "
                f"AUC {mean['AUC']:.2f}±{std['AUC']:.2f}   "
                f"准确率 {mean['准确率']:.2f}"
            )

            fpr, tpr, _ = micro_roc(y_true, y_prob)
            curves[model_name] = (fpr, tpr, auc(fpr, tpr))
            cm = plot_confusion(y_true, y_pred, model_name, arm_name)
            print("  10折汇总混淆矩阵 (行=真实, 列=预测):")
            for i, row_ in enumerate(cm):
                print(f"    {LABEL_NAMES[i]:>9s}  {row_}")

            summary_rows.append({
                "特征方案": arm_name,
                "建模方法": model_name,
                "精确率": f"{mean['精确率']:.2f}±{std['精确率']:.2f}",
                "召回率": f"{mean['召回率']:.2f}±{std['召回率']:.2f}",
                "F1值": f"{mean['F1值']:.2f}±{std['F1值']:.2f}",
                "AUC平均值": f"{mean['AUC']:.2f}±{std['AUC']:.2f}",
                "准确率": f"{mean['准确率']:.2f}±{std['准确率']:.2f}",
            })
            fold_df.to_csv(
                RESULTS_DIR
                / f"perfold_{SCHEME_SLUG[arm_name]}_{MODEL_SLUG[model_name]}.csv",
                index=False, encoding="utf-8-sig",
            )

        print(f"\n  → ROC 曲线已保存: {plot_roc(curves, arm_name).name}\n")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        RESULTS_DIR / "metrics_summary.csv", index=False, encoding="utf-8-sig"
    )

    print("=" * 74)
    print("结果汇总（对应论文表 4）")
    print("=" * 74)
    print(summary.to_string(index=False))
    print()
    print(f"详细结果已保存至: {RESULTS_DIR}")
    print("=" * 74)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="复现论文步骤 7-8：模型训练与评估"
    )
    parser.add_argument(
        "--features",
        default="features_138.csv",
        help="data/ 下的特征表文件名。"
             "features_138.csv → 论文三方案对照（几何10/深度128/融合138）；"
             "geometric_features_61.csv → 61 维几何特征；"
             "geometric_features.csv → 30 维几何特征",
    )
    args = parser.parse_args()

    # 含 e0..e127 深度列 → 走论文 2.3 节的三方案对照
    _head = pd.read_csv(DATA / args.features, encoding="utf-8-sig", nrows=1)
    if any(is_deep_col(c) for c in _head.columns):
        main_fusion(args.features)
    else:
        main(args.features)
