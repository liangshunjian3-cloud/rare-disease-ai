# -*- coding: utf-8 -*-
"""用 61 维语义几何辅助模型解释主模型预测类别的几何证据。"""

import numpy as np

from extract_features_61 import FEATURE_LABELS_CN


def _select_class_values(values, class_index, n_features, n_classes):
    """兼容 SHAP 多输出 API 的 list 与 ndarray 返回形式。"""
    if isinstance(values, list):
        if class_index >= len(values):
            raise RuntimeError("SHAP 返回的类别数少于模型类别数")
        selected = np.asarray(values[class_index], dtype=float).reshape(-1)
    else:
        array = np.asarray(values, dtype=float)
        if array.shape == (1, n_features, n_classes):
            selected = array[0, :, class_index]
        elif array.shape == (1, n_classes, n_features):
            selected = array[0, class_index, :]
        elif n_classes == 1 and array.shape == (1, n_features):
            selected = array[0]
        else:
            raise RuntimeError(f"无法识别 SHAP 多分类输出形状: {array.shape}")

    if selected.shape != (n_features,):
        raise RuntimeError(
            f"SHAP 特征贡献数为 {selected.size}，模型特征数为 {n_features}"
        )
    return selected


def explain_features(features, bundle, class_index, top_n=5):
    """解释几何辅助模型中指定类别概率相对背景样本基准值的贡献。

    `features` 和 bundle 中的背景样本均为未标准化的几何特征；
    标准化在 SHAP 调用模型时通过 bundle 中的 scaler 完成。
    这不是主深度模型的 SHAP 归因。
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "启用 SHAP 解释需要安装 shap：pip install shap"
        ) from exc

    feature_names = bundle["feature_cols"]
    background = np.asarray(bundle["shap_background"], dtype=float)
    sample = np.asarray(features, dtype=float).reshape(1, -1)
    n_features = len(feature_names)
    n_classes = len(bundle["label_names"])

    if sample.shape[1] != n_features:
        raise RuntimeError(
            f"待解释特征数为 {sample.shape[1]}，模型要求 {n_features}"
        )
    if background.ndim != 2 or background.shape[1] != n_features:
        raise RuntimeError("模型中的 SHAP 背景样本维度与特征列不匹配")
    if not np.isfinite(sample).all() or not np.isfinite(background).all():
        raise RuntimeError("待解释特征或 SHAP 背景样本中存在 NaN 或无穷值")

    def predict_proba(raw_features):
        scaled = bundle["scaler"].transform(np.asarray(raw_features, dtype=float))
        return bundle["model"].predict_proba(scaled)

    explainer = shap.KernelExplainer(predict_proba, background)
    shap_values = explainer.shap_values(
        sample, nsamples="auto", l1_reg=0
    )
    contributions = _select_class_values(
        shap_values, class_index, n_features, n_classes
    )

    expected_values = np.asarray(explainer.expected_value, dtype=float).reshape(-1)
    if class_index >= len(expected_values):
        raise RuntimeError("SHAP 基准值的类别数少于模型类别数")

    probabilities = predict_proba(sample)[0]
    geometry_class_index = int(np.argmax(probabilities))
    positive_indices = np.flatnonzero(contributions > 0)
    ranked = positive_indices[
        np.argsort(contributions[positive_indices])[::-1][:top_n]
    ]
    features_out = []
    for idx in ranked:
        value = float(contributions[idx])
        name = feature_names[idx]
        features_out.append({
            "name": name,
            "label_cn": FEATURE_LABELS_CN.get(name, name),
            "value": float(sample[0, idx]),
            "shap_value": value,
            "effect": "推高该类概率" if value > 0 else "压低该类概率",
        })

    return {
        "class": bundle["label_names"][class_index],
        "class_cn": bundle["label_names_cn"][class_index],
        "explainer_model": "61维几何辅助分类器",
        "geometry_model_prediction": bundle["label_names"][geometry_class_index],
        "geometry_model_prediction_cn": bundle["label_names_cn"][geometry_class_index],
        "geometry_model_prediction_probability": float(
            probabilities[geometry_class_index]
        ),
        "base_value": float(expected_values[class_index]),
        "predicted_probability": float(probabilities[class_index]),
        "features": features_out,
    }
