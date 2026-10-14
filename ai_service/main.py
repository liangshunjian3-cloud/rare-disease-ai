from functools import lru_cache
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

RECOGNIZE_SYSTEM_ROOT = Path(__file__).resolve().parent.parent / "recognize_system"
PREDICT_SRC = RECOGNIZE_SYSTEM_ROOT / "src"
if str(PREDICT_SRC) not in sys.path:
    sys.path.insert(0, str(PREDICT_SRC))

from predict import load_model, predict_image  # noqa: E402

app = FastAPI(title="罕见病辅助识别系统")

ALLOWED_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}
MAX_IMAGE_SIZE = 10 * 1024 * 1024


@lru_cache(maxsize=1)
def get_model_bundle():
    """在当前服务进程中只加载一次主预测模型及 dlib 模型。"""
    return load_model()


@app.get("/health")
def health():
    """检查后端服务是否正常。"""
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    explain: bool = Form(True),
    threshold: float | None = Form(default=None, ge=0, le=1),
):
    """预测上传的人脸图像；默认附带几何辅助模型的 SHAP 特征解释。"""
    suffix = ALLOWED_TYPES.get(file.content_type)
    if suffix is None:
        raise HTTPException(
            status_code=400,
            detail="请上传 JPG、PNG 或 WebP 图片",
        )

    content = await file.read(MAX_IMAGE_SIZE + 1)
    if not content:
        raise HTTPException(status_code=400, detail="上传的图片为空")
    if len(content) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=413, detail="图片不能超过 10 MB")

    temp_path = None
    try:
        with NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_file.write(content)
            temp_path = Path(temp_file.name)

        try:
            bundle = await run_in_threadpool(get_model_bundle)
            result = await run_in_threadpool(
                predict_image, temp_path, bundle, threshold, explain
            )
        except (FileNotFoundError, ImportError, RuntimeError) as exc:
            raise HTTPException(
                status_code=503,
                detail=f"模型暂不可用：{exc}",
            ) from exc

        if result.error:
            raise HTTPException(status_code=422, detail=result.error)

        label_names_cn = dict(zip(bundle["label_names"], bundle["label_names_cn"]))
        top_k = [
            {
                "label": label_names_cn[name],
                "labelEn": name,
                "probability": probability,
            }
            for name, probability in sorted(
                result.probs.items(), key=lambda item: item[1], reverse=True
            )
        ]

        explanation = result.explanation
        explanations = []
        if explanation:
            explanations = [
                {
                    "name": item["label_cn"],
                    "feature": item["name"],
                    "value": item["value"],
                    "shapValue": item["shap_value"],
                    "effect": item["effect"],
                }
                for item in explanation["features"]
            ]

        return {
            "status": "rejected" if result.rejected else "ok",
            "prediction": {
                "label": result.label_cn or "不确定",
                "labelEn": result.label,
                "probability": result.confidence,
                "rejected": result.rejected,
                "threshold": result.threshold,
            },
            "topK": top_k,
            "probabilities": result.probs,
            "explanation": (
                {
                    "class": explanation["class"],
                    "classCn": explanation["class_cn"],
                    "model": explanation["explainer_model"],
                    "geometryModelPrediction": explanation[
                        "geometry_model_prediction"
                    ],
                    "geometryModelPredictionCn": explanation[
                        "geometry_model_prediction_cn"
                    ],
                    "geometryModelPredictionProbability": explanation[
                        "geometry_model_prediction_probability"
                    ],
                    "baseValue": explanation["base_value"],
                    "predictedProbability": explanation["predicted_probability"],
                }
                if explanation
                else None
            ),
            "explanations": explanations,
            "warning": (
                "概率仅表示三个候选类别之间的模型输出，不代表患病概率；"
                "SHAP 特征来自几何辅助模型，不是深度主模型的直接归因，"
                "也不能用于临床诊断。"
            ),
        }
    finally:
        await file.close()
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
