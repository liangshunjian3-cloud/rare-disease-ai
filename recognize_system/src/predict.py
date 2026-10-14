# -*- coding: utf-8 -*-
"""
模型推理：一张人脸图片 → 三种罕见病之一（或「不确定」）。

命令行
------
    python src/predict.py face.jpg                    # 单张
    python src/predict.py face.jpg --threshold 0.7    # 调高拒识阈值（更保守）
    python src/predict.py data/raw/down/ -r           # 递归批量
    python src/predict.py face.jpg --json             # 机器可读输出

Python 接口
-----------
    from src.predict import load_model, predict_image

    bundle = load_model()
    r = predict_image("face.jpg", bundle)
    print(r.label_cn, r.confidence)       # 唐氏综合征 0.87
    print(r.probs)                        # {'DiGeorge': 0.05, 'Down': 0.87, ...}
    r = predict_image("face.jpg", bundle, explain=True)
    print(r.explanation)                  # 几何辅助模型对主预测类别的贡献
    if r.rejected:                        # 置信度不足，未给出病种
        ...

拒识机制
--------
最高类概率低于阈值时，`rejected=True` 且不输出病种，只提示人工复核。
这么做是因为模型只见过 3 个病种，碰到不属于这三类的面孔时仍会强行归类；
阈值越高越保守（见 models/final/threshold_table.csv）。

⚠️ 本模型样本量仅 117（3 类），无独立测试集，**不可用于任何临床诊断**。
"""

from pathlib import Path
import argparse
import json
import sys
from dataclasses import dataclass, field, asdict

import joblib
import numpy as np

# 允许 `python src/predict.py` 与 `from src.predict import ...` 两种用法
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from face_embedding import (  # noqa: E402
    FaceError,
    extract_descriptor,
    load_models,
)
from extract_features_61 import (  # noqa: E402
    FEATURE_NAMES,
    feature_vector_from_landmarks,
)
from explain import explain_features  # noqa: E402

ROOT = _SRC_DIR.parent
DEFAULT_MODEL = ROOT / "models" / "final" / "svm_deep.joblib"
DEFAULT_GEOMETRY_MODEL = ROOT / "models" / "final" / "svm_geo61.joblib"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ==========================================================================
# 结果对象
# ==========================================================================
@dataclass
class Prediction:
    """单张图片的预测结果。"""

    image_path: str
    label: str | None = None          # 英文名；拒识或出错时为 None
    label_cn: str | None = None       # 中文名
    confidence: float = 0.0           # 最高类概率
    probs: dict = field(default_factory=dict)
    rejected: bool = False            # 置信度低于阈值
    threshold: float | None = None    # 本次判定实际使用的拒识阈值
    explanation: dict | None = None   # 可选：几何辅助模型对主模型类别的 SHAP 贡献
    error: str | None = None          # 处理失败的原因（如未检测到人脸）

    @property
    def ok(self) -> bool:
        """是否成功给出了一个病种判断。"""
        return self.error is None and not self.rejected and self.label is not None

    def to_dict(self) -> dict:
        return asdict(self)


# ==========================================================================
# 模型加载
# ==========================================================================
def load_model(model_path=None, load_face_models=True, geometry_model_path=None):
    """载入推理所需的全部东西。

    返回 bundle 字典，除主深度模型外还带上 dlib 的人脸模型（键 `_dlib`）。
    几何解释模型仅在请求解释时加载。
    dlib 模型只在进程内加载一次，批量推理时请复用同一个 bundle。
    """
    model_path = Path(model_path or DEFAULT_MODEL)
    if not model_path.exists():
        raise FileNotFoundError(
            f"未找到模型文件: {model_path}\n请先运行: python src/train_final.py"
        )

    bundle = joblib.load(model_path)
    for key in (
        "model", "scaler", "feature_cols", "label_names",
        "label_names_cn", "default_threshold",
    ):
        if key not in bundle:
            raise RuntimeError(f"模型文件缺少必需字段 `{key}`，可能已损坏或是旧版本")
    if len(bundle["feature_cols"]) != 128:
        raise RuntimeError(
            "主模型不是预期的 128 维 dlib 模型，请指定正确的模型文件"
        )
    expected_classes = list(range(len(bundle["label_names"])))
    if list(bundle["model"].classes_) != expected_classes:
        raise RuntimeError("主模型内部类别顺序与标签名称不匹配")

    if load_face_models:
        bundle["_dlib"] = load_models()
    bundle["_geometry_model_path"] = Path(
        geometry_model_path or DEFAULT_GEOMETRY_MODEL
    )

    return bundle


def load_geometry_model(bundle):
    """加载并校验只用于辅助解释的 61 维几何模型。"""
    if "_geometry_model" in bundle:
        return bundle["_geometry_model"]

    model_path = Path(bundle.get("_geometry_model_path", DEFAULT_GEOMETRY_MODEL))
    if not model_path.exists():
        raise FileNotFoundError(
            f"未找到几何辅助模型: {model_path}\n"
            "请先运行: python src/train_geometry.py"
        )

    geometry_bundle = joblib.load(model_path)
    required = (
        "model", "scaler", "feature_cols", "shap_background",
        "label_names", "label_names_cn",
    )
    missing = [key for key in required if key not in geometry_bundle]
    if missing:
        raise RuntimeError(
            f"几何模型缺少必需字段 {missing}，请运行 src/train_geometry.py 重新生成"
        )
    if geometry_bundle["feature_cols"] != list(FEATURE_NAMES):
        raise RuntimeError(
            "几何模型特征定义不匹配，请运行 src/train_geometry.py 重新训练"
        )
    if geometry_bundle["label_names"] != bundle["label_names"]:
        raise RuntimeError("主模型与几何辅助模型的类别顺序不一致")
    if list(geometry_bundle["model"].classes_) != list(
        range(len(geometry_bundle["label_names"]))
    ):
        raise RuntimeError("几何辅助模型内部类别顺序与标签名称不匹配")

    bundle["_geometry_model"] = geometry_bundle
    return geometry_bundle


# ==========================================================================
# 推理
# ==========================================================================
def predict_image(img_path, bundle, threshold=None, explain=False):
    """对单张图片预测。

    bundle   : load_model() 的返回值
    threshold: 拒识阈值；None 则用模型自带默认值
    explain  : 是否用几何辅助模型解释主模型的最高概率类别（需安装 shap）
    """
    threshold = bundle["default_threshold"] if threshold is None else threshold
    names = bundle["label_names"]
    names_cn = bundle["label_names_cn"]

    try:
        desc, landmarks = extract_descriptor(img_path, bundle["_dlib"])
    except FaceError as exc:
        return Prediction(image_path=str(img_path), error=str(exc))
    except Exception as exc:  # noqa: BLE001 — 坏图/格式异常都归到这里
        return Prediction(image_path=str(img_path), error=f"{type(exc).__name__}: {exc}")

    # 特征顺序必须与训练时一致 —— bundle 里存了列名，这里靠位置对齐
    if len(desc) != len(bundle["feature_cols"]):
        return Prediction(
            image_path=str(img_path),
            error=f"描述子维度 {len(desc)} 与模型期望的 "
                  f"{len(bundle['feature_cols'])} 不符",
        )

    x = bundle["scaler"].transform(desc.reshape(1, -1))
    proba = bundle["model"].predict_proba(x)[0]

    probs = {n: float(p) for n, p in zip(names, proba)}
    top = int(np.argmax(proba))
    confidence = float(proba[top])
    rejected = confidence < threshold
    explanation = None
    if explain:
        geometry_bundle = load_geometry_model(bundle)
        geometric_features = feature_vector_from_landmarks(landmarks)
        explanation = explain_features(geometric_features, geometry_bundle, top)

    return Prediction(
        image_path=str(img_path),
        label=None if rejected else names[top],
        label_cn=None if rejected else names_cn[top],
        confidence=confidence,
        probs=probs,
        rejected=rejected,
        threshold=float(threshold),
        explanation=explanation,
    )


def collect_images(path, recursive=True):
    """把一个路径展开成图片列表。目录则按后缀过滤。"""
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"路径不存在: {path}")

    it = path.rglob("*") if recursive else path.glob("*")
    return sorted(p for p in it if p.suffix.lower() in IMAGE_SUFFIXES)


def predict_batch(paths, bundle, threshold=None, verbose=False, explain=False):
    """对一组路径预测，返回 Prediction 列表。"""
    results = []
    for i, p in enumerate(paths, 1):
        results.append(predict_image(p, bundle, threshold, explain=explain))
        if verbose and i % 20 == 0:
            print(f"  已处理 {i}/{len(paths)}", file=sys.stderr)
    return results


# ==========================================================================
# 输出格式化
# ==========================================================================
BAR_WIDTH = 12


def format_single(r: Prediction, bundle):
    """单张图片的可读输出。"""
    lines = [f"图片: {r.image_path}", ""]

    if r.error:
        lines += [f"  ✗ 处理失败: {r.error}", ""]
        return "\n".join(lines)

    if r.rejected:
        thr = r.threshold if r.threshold is not None else bundle["default_threshold"]
        lines += [
            f"  ⚠ 不确定：最高置信度 {r.confidence:.2f} < 阈值 {thr:.2f}",
            "     建议人工复核，不输出病种",
            "",
        ]
    else:
        lines += [
            f"  预测病种 : {r.label_cn}（{r.label}）",
            f"  置信度   : {r.confidence:.2f}",
            "",
        ]

    if r.explanation:
        lines.append(
            f"  几何辅助模型 SHAP（针对主预测类别 {r.explanation['class_cn']}，"
            f"相对基准 {r.explanation['base_value']:.3f}）："
        )
        lines.append(
            f"    几何模型自身预测: {r.explanation['geometry_model_prediction_cn']} "
            f"({r.explanation['geometry_model_prediction_probability']:.3f})"
        )
        for item in r.explanation["features"]:
            lines.append(
                f"    {item['label_cn']}: {item['value']:.3f}  "
                f"{item['effect']} ({item['shap_value']:+.4f})"
            )
        lines.append(
            "  注：这是几何辅助模型的解释，不是深度主模型的直接归因，"
            "也不是因果或诊断依据。"
        )
        lines.append("")

    lines.append("  各类概率 :")
    for name, name_cn, p in zip(bundle["label_names"],
                                bundle["label_names_cn"],
                                (r.probs[n] for n in bundle["label_names"])):
        bar = "█" * int(round(p * BAR_WIDTH))
        lines.append(f"    {name_cn:<16s} {p:.3f}  {bar}")
    lines.append("")
    return "\n".join(lines)


def format_batch(results, bundle):
    """批量结果的可读汇总。"""
    lines = []

    n_err = sum(1 for r in results if r.error)
    n_rej = sum(1 for r in results if r.ok is False and r.error is None)
    n_ok = sum(1 for r in results if r.ok)

    lines.append(f"共 {len(results)} 张：成功判定 {n_ok}，"
                 f"拒识 {n_rej}，失败 {n_err}")
    lines.append("")
    lines.append(f"  {'状态':<8}{'置信度':>8}  {'结果':<22} 图片")
    lines.append("  " + "-" * 74)

    for r in results:
        if r.error:
            status, conf, label = "失败", "-", r.error[:40]
        elif r.rejected:
            status, conf, label = "拒识", f"{r.confidence:.2f}", "（置信度不足）"
        else:
            status, conf, label = "成功", f"{r.confidence:.2f}", r.label_cn

        name = Path(r.image_path).name
        lines.append(f"  {status:<8}{conf:>8}  {label:<22} {name}")

    # 判定成功者的病种分布
    if n_ok:
        from collections import Counter
        counter = Counter(r.label_cn for r in results if r.ok)
        lines.append("")
        lines.append("  判定结果的病种分布：")
        for label, cnt in counter.most_common():
            lines.append(f"    {label:<20s} {cnt:>4d}")

    lines.append("")
    return "\n".join(lines)


# ==========================================================================
# 命令行
# ==========================================================================
def build_parser():
    p = argparse.ArgumentParser(
        description="罕见病面部识别模型推理（3 分类）",
        epilog="⚠️ 本模型样本量仅 117，不可用于任何临床诊断。",
    )
    p.add_argument("path", help="图片路径，或目录（配合 -r）")
    p.add_argument("-m", "--model", default=None,
                   help=f"模型文件路径（默认 {DEFAULT_MODEL.relative_to(ROOT)}）")
    p.add_argument("-t", "--threshold", type=float, default=None,
                   help="拒识阈值，低于它则输出「不确定」；越大越保守")
    p.add_argument("-r", "--recursive", action="store_true",
                   help="路径是目录时递归处理其中的所有图片")
    p.add_argument("--json", action="store_true", help="输出 JSON，便于程序处理")
    p.add_argument(
        "--explain", action="store_true",
        help="用几何辅助模型的 SHAP 解释主预测类别（需 pip install shap，计算较慢）",
    )
    p.add_argument(
        "--geometry-model", default=None,
        help=f"几何辅助模型文件（默认 {DEFAULT_GEOMETRY_MODEL.relative_to(ROOT)}）",
    )
    p.add_argument("-q", "--quiet", action="store_true",
                   help="批量模式下只输出汇总")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.explain:
        try:
            import shap  # noqa: F401
        except ImportError:
            print("错误: --explain 需要安装 SHAP：pip install shap", file=sys.stderr)
            return 2

    # 目录必须显式加 -r，避免误把整个 data/ 拖进来
    path = Path(args.path)
    if path.is_dir() and not args.recursive:
        build_parser().error(f"{path} 是目录，请加 -r 递归处理")

    try:
        bundle = load_model(args.model, geometry_model_path=args.geometry_model)
        if args.explain:
            load_geometry_model(bundle)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    threshold = args.threshold
    if threshold is None:
        threshold = bundle["default_threshold"]

    try:
        paths = collect_images(path, recursive=args.recursive)
    except FileNotFoundError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    if not paths:
        print(f"错误: {path} 下没有找到图片（支持 {'/'.join(sorted(IMAGE_SUFFIXES))}）",
              file=sys.stderr)
        return 2

    single = len(paths) == 1 and path.is_file()
    results = predict_batch(paths, bundle, threshold,
                            verbose=len(paths) > 20 and not args.json,
                            explain=args.explain)

    if args.json:
        payload = {
            "model": {
                "cv_metrics": bundle.get("cv_metrics"),
                "n_samples": bundle.get("n_samples"),
                "trained_at": bundle.get("trained_at"),
            },
            "prediction_model": "128维 dlib 深度分类器",
            "explanation_model": (
                "61维几何辅助分类器" if args.explain else None
            ),
            "threshold": threshold,
            "results": [r.to_dict() for r in results],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if single:
        print()
        print(format_single(results[0], bundle))
    elif args.quiet:
        print(format_batch(results, bundle))
    else:
        for r in results:
            print(format_single(r, bundle))
        print(format_batch(results, bundle))

    # 有失败样本时返回非零，便于脚本判断
    return 1 if any(r.error for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
