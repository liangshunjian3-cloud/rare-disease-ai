# -*- coding: utf-8 -*-
"""训练 61 维几何辅助模型，供主预测结果的 SHAP 辅助解释使用。"""

from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from extract_features_61 import FEATURE_NAMES, feature_vector_from_landmarks

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LANDMARKS_FILE = DATA / "landmarks_68_raw.csv"
OUT_DIR = ROOT / "models" / "final"
OUT_FILE = OUT_DIR / "svm_geo61.joblib"

SEED = 42
LABEL_MAP = {"digeorge": 0, "down": 1, "williams": 2}
LABEL_NAMES = ["DiGeorge", "Down", "Williams"]
LABEL_NAMES_CN = ["DiGeorge 综合征", "唐氏综合征", "Williams 综合征"]
SVM_PARAMS = dict(
    kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=SEED
)


def load_features():
    """从原图坐标系关键点提取与线上推理一致的 61 维特征。"""
    if not LANDMARKS_FILE.exists():
        raise FileNotFoundError(
            f"未找到原图坐标系关键点: {LANDMARKS_FILE}\n"
            "请先运行 src/extract_embeddings_128.py 生成 landmarks_68_raw.csv"
        )

    records = pd.read_csv(LANDMARKS_FILE, encoding="utf-8-sig")
    point_cols = [f"{axis}{i}" for i in range(68) for axis in ("x", "y")]
    missing = [name for name in point_cols if name not in records.columns]
    if missing:
        raise RuntimeError(f"{LANDMARKS_FILE.name} 缺少关键点列: {missing[:5]}")

    points = records[point_cols].to_numpy(dtype=float).reshape(-1, 68, 2)
    X = np.vstack([feature_vector_from_landmarks(row) for row in points])
    if X.shape != (len(records), len(FEATURE_NAMES)):
        raise RuntimeError(
            f"几何特征矩阵形状为 {X.shape}，"
            f"应为 {(len(records), len(FEATURE_NAMES))}"
        )
    if not np.isfinite(X).all():
        raise RuntimeError("几何特征矩阵中存在 NaN 或无穷值")

    y = records["label"].map(LABEL_MAP).to_numpy()
    if np.isnan(y).any():
        raise RuntimeError("存在无法映射的标签")
    return records, X, y.astype(int)


def make_shap_background(X, y, per_class=10):
    """按类别均衡抽取背景样本，作为 SHAP 的参考人群。"""
    rng = np.random.default_rng(SEED)
    selected = []
    for class_id, name in enumerate(LABEL_NAMES):
        class_indices = np.flatnonzero(y == class_id)
        if not len(class_indices):
            raise RuntimeError(f"训练数据缺少类别 {name}")
        count = min(per_class, len(class_indices))
        selected.extend(rng.choice(class_indices, size=count, replace=False))
    return X[np.asarray(selected, dtype=int)]


def oof_probabilities(X, y, groups, n_splits=10):
    """按受试者分组交叉验证，只在训练折内拟合标准化器。"""
    oof = np.zeros((len(y), len(LABEL_NAMES)))
    is_oof = np.zeros(len(y), dtype=bool)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    for train_indices, test_indices in cv.split(X, y, groups):
        scaler = StandardScaler().fit(X[train_indices])
        model = SVC(**SVM_PARAMS).fit(
            scaler.transform(X[train_indices]), y[train_indices]
        )
        if list(model.classes_) != list(range(len(LABEL_NAMES))):
            raise RuntimeError(f"交叉验证折的类别不完整: {model.classes_}")
        oof[test_indices] = model.predict_proba(
            scaler.transform(X[test_indices])
        )
        is_oof[test_indices] = True

    if not is_oof.all():
        raise RuntimeError("交叉验证没有为每条训练记录生成 out-of-fold 预测")
    return oof


def main():
    records, X, y = load_features()
    groups = records["subject_id"].to_numpy()
    oof = oof_probabilities(X, y, groups)
    cv_metrics = {
        "auc_micro": float(
            roc_auc_score(y, oof, multi_class="ovr", average="micro")
        ),
        "accuracy": float(accuracy_score(y, oof.argmax(axis=1))),
        "f1_macro": float(f1_score(y, oof.argmax(axis=1), average="macro")),
    }

    scaler = StandardScaler().fit(X)
    model = SVC(**SVM_PARAMS).fit(scaler.transform(X), y)
    bundle = {
        "model": model,
        "scaler": scaler,
        "feature_cols": list(FEATURE_NAMES),
        "shap_background": make_shap_background(X, y),
        "label_map": LABEL_MAP,
        "label_names": LABEL_NAMES,
        "label_names_cn": LABEL_NAMES_CN,
        "cv_metrics": cv_metrics,
        "cv_protocol": "StratifiedGroupKFold(n_splits=10)，按 subject_id 分组",
        "model_role": "辅助几何解释模型；不替代 128 维主预测模型",
        "n_samples": int(len(y)),
        "n_subjects": int(len(np.unique(groups))),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, OUT_FILE)
    print(f"辅助几何模型已保存: {OUT_FILE.relative_to(ROOT)}")
    print(f"训练样本: {len(y)}   几何特征: {X.shape[1]} 维")
    print(
        "分组交叉验证参考指标（不是独立测试集成绩）: "
        f"AUC {cv_metrics['auc_micro']:.3f}   "
        f"准确率 {cv_metrics['accuracy']:.3f}   "
        f"macro F1 {cv_metrics['f1_macro']:.3f}"
    )
    print("下一步: python src/predict.py <图片路径> --explain")
    return bundle


if __name__ == "__main__":
    main()
