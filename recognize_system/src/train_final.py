# -*- coding: utf-8 -*-
"""
在全部样本上训练最终模型并序列化，供 src/predict.py 推理使用。

特征：128 维 dlib 描述子（data/embeddings_128.csv）
模型：SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)
输出：models/final/svm_deep.joblib

⚠️ 关于「模型效果」的口径 —— 这点很容易搞混：

    CSV 中记录的指标是【10 折交叉验证对"这套流程"的估计】，
    不是这个 .joblib 文件本身的成绩。交叉验证里每个折的模型都只在 90% 数据上
    训练过、且训练完就丢了；而这里保存的模型是在【全部 117 个样本】上训练的，
    它的真实表现没有任何独立数据可以衡量。

    换句话说：这个模型在训练集上的表现只会比 0.94 更好看，而那个数字没有意义。
    要评估它，必须另找一批从未参与训练的数据。

同时输出一张「阈值 → 拒识比例 → 拒识后准确率」对照表（基于 out-of-fold 概率），
给 predict.py 的默认拒识阈值提供依据。
"""

from pathlib import Path
import argparse
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT_DIR = ROOT / "models" / "final"
OUT_FILE = OUT_DIR / "svm_deep.joblib"

SEED = 42

# 论文 2.1 节的类别编码
LABEL_MAP = {"digeorge": 0, "down": 1, "williams": 2}
LABEL_NAMES = ["DiGeorge", "Down", "Williams"]
LABEL_NAMES_CN = ["DiGeorge 综合征", "唐氏综合征", "Williams 综合征"]

# 与 reproduce_paper.py 保持一致的模型配置
SVM_PARAMS = dict(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=SEED)

# 默认拒识阈值。见 train_final.py 末尾打印的对照表。
DEFAULT_THRESHOLD = 0.60


def load_features():
    """读入 128 维深度特征，返回 (数据, 特征名, X, y)。"""
    emb = pd.read_csv(DATA / "embeddings_128.csv", encoding="utf-8-sig")
    cols = [c for c in emb.columns if c.startswith("e") and c[1:].isdigit()]
    if len(cols) != 128:
        raise RuntimeError(f"深度特征列数 {len(cols)}，应为 128")

    X = emb[cols].to_numpy(dtype=float)
    if np.isnan(X).any():
        bad = emb.loc[np.isnan(X).any(axis=1), "image_id"].tolist()
        raise RuntimeError(f"特征矩阵含 NaN，涉及 {len(bad)} 张图: {bad[:5]}")

    y = emb["label"].map(LABEL_MAP).to_numpy()
    if np.isnan(y).any():
        raise RuntimeError("存在无法映射的标签")

    return emb, cols, X, y


def oof_probabilities(X, y, groups, n_splits=10):
    """out-of-fold 概率，用于评估阈值策略。

    按受试者分组划分（同一人不跨折），与 reproduce_paper.py 的口径一致。
    """
    n_class = len(LABEL_NAMES)
    oof = np.zeros((len(y), n_class))
    is_oof = np.zeros(len(y), dtype=bool)

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for tr, te in cv.split(X, y, groups):
        scaler = StandardScaler().fit(X[tr])
        model = SVC(**SVM_PARAMS).fit(scaler.transform(X[tr]), y[tr])
        assert list(model.classes_) == list(range(n_class)), model.classes_
        oof[te] = model.predict_proba(scaler.transform(X[te]))
        is_oof[te] = True

    return oof, is_oof


def threshold_table(oof, y, thresholds):
    """对每个阈值，算出拒识比例与拒识后的准确率。"""
    top_prob = oof.max(axis=1)
    pred = oof.argmax(axis=1)

    rows = []
    for t in thresholds:
        keep = top_prob >= t
        kept = int(keep.sum())
        rows.append({
            "阈值": t,
            "拒识比例": 1 - kept / len(y),
            "保留样本": kept,
            "拒识后准确率": accuracy_score(y[keep], pred[keep]) if kept else float("nan"),
            "拒识样本中被判错的比例": (
                float((pred[~keep] != y[~keep]).mean()) if (~keep).any() else float("nan")
            ),
        })
    return pd.DataFrame(rows)


def main(threshold=DEFAULT_THRESHOLD):
    emb, cols, X, y = load_features()
    groups = emb["subject_id"].to_numpy()

    print("=" * 70)
    print("训练最终模型")
    print("=" * 70)
    print(f"样本数     : {len(y)}")
    print(f"受试者数   : {len(np.unique(groups))}")
    print(f"特征维度   : {X.shape[1]}  (128 维 dlib 描述子)")
    print(f"类别分布   : {dict(zip(LABEL_NAMES, np.bincount(y)))}")
    print(f"模型       : SVC(kernel=rbf, C=1.0, gamma='scale')")
    print()

    # ---- 1) 交叉验证：评估「这套流程」，不是在评估下面保存的那个模型 ----
    oof, is_oof = oof_probabilities(X, y, groups)
    assert is_oof.all()

    acc = accuracy_score(y, oof.argmax(axis=1))
    f1 = f1_score(y, oof.argmax(axis=1), average="macro")
    auc = roc_auc_score(y, oof, multi_class="ovr", average="micro")
    print("【参考指标】10 折交叉验证（按受试者分组）对「这套流程」的估计：")
    print(f"  AUC {auc:.3f}   准确率 {acc:.3f}   macro F1 {f1:.3f}")
    print("  ⚠️ 这是对流程的估计，不是下面保存的那个模型本身的成绩。")
    print()

    # ---- 2) 阈值对照表 ----
    ttable = threshold_table(oof, y, [0.40, 0.50, 0.60, 0.70, 0.80])
    print("【拒识阈值对照】基于 out-of-fold 概率：")
    print(ttable.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()

    row = ttable[ttable["阈值"] == threshold]
    if len(row):
        r = row.iloc[0]
        print(f"  默认阈值 {threshold}：拒识 {r['拒识比例']:.1%} 的样本，"
              f"保留部分的准确率 {r['拒识后准确率']:.3f}")
        print("  （阈值越高越保守：拒识更多、但留下的更可信）")
    print()

    # ---- 3) 在全部样本上训练最终模型 ----
    scaler = StandardScaler().fit(X)
    model = SVC(**SVM_PARAMS).fit(scaler.transform(X), y)
    train_acc = accuracy_score(y, model.predict(scaler.transform(X)))
    print(f"最终模型已在全部 {len(y)} 个样本上训练完毕")
    print(f"  训练集内准确率 {train_acc:.3f}  ← 无意义，仅说明拟合没崩，不要对外引用")
    print()

    # ---- 4) 打包保存 ----
    # scaler 与特征列顺序必须跟模型一起存：推理时要用同一套标准化参数、
    # 按同样顺序取列，否则预测会静默出错（不报错，但结果是错的）。
    bundle = {
        "model": model,
        "scaler": scaler,
        "feature_cols": cols,
        "label_map": LABEL_MAP,
        "label_names": LABEL_NAMES,
        "label_names_cn": LABEL_NAMES_CN,
        "default_threshold": threshold,
        "cv_metrics": {"auc_micro": float(auc), "accuracy": float(acc),
                       "f1_macro": float(f1)},
        "cv_protocol": "StratifiedGroupKFold(n_splits=10)，按 subject_id 分组",
        "n_samples": int(len(y)),
        "n_subjects": int(len(np.unique(groups))),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "notes": (
            "在全部样本上训练，无独立测试集。cv_metrics 是对「这套流程」的"
            "交叉验证估计，不能当作本模型自身的成绩。"
        ),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, OUT_FILE)
    size_mb = OUT_FILE.stat().st_size / 1024 / 1024
    print(f"已保存: {OUT_FILE.relative_to(ROOT)}  ({size_mb:.2f} MB)")
    print("  bundle 内容: model / scaler / 128维特征列 / 标签映射 / 参考指标")
    print()
    print(f"下一步: python src/predict.py <图片路径>")
    print("=" * 70)

    ttable.to_csv(OUT_DIR / "threshold_table.csv", index=False, encoding="utf-8-sig")
    return bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练并保存最终推理模型")
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"默认拒识阈值，低于它则输出「不确定」（默认 {DEFAULT_THRESHOLD}）",
    )
    args = parser.parse_args()
    main(args.threshold)
