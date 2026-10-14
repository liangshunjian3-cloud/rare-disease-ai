# -*- coding: utf-8 -*-
"""
Dlib 人脸嵌入的共享底层：一张图片 → 128 维描述子。

`extract_embeddings_128.py`（批量提取）与 `predict.py`（单张推理）都调用这里。
抽成公共模块是为了让「训练时用的特征」和「推理时算的特征」出自同一份代码 ——
描述子对对齐方式很敏感（实测：检测框平移 5px，描述子就变化 0.043），
两处参数一旦漂移，模型看到的就是与训练不同尺度的特征，预测会静默出错。

三个模型的分工：
    1. 检测器         图片 → 人脸方框（只定位，不认人）
    2. 关键点模型     图片 + 方框 → 68 个坐标点
    3. 描述子模型     图片 + 68 点 → 128 维向量

为什么关键点必须传给第 3 步：
    描述子模型内部会挑 5 个点（两眼中点、鼻尖、两嘴角）做仿射对齐，
    把人脸摆正、缩放到标准姿态后再送进 ResNet。只给检测框会明显变差。
"""

from pathlib import Path

import dlib
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = ROOT / "models"

PREDICTOR_FILE = "shape_predictor_68_face_landmarks.dat"
FACEREC_FILE = "dlib_face_recognition_resnet_model_v1.dat"

N_LANDMARKS = 68
N_EMBEDDING = 128

# ---------------------------------------------------------------- 参数
# ⚠️ 这两个值必须与训练时完全一致！改动会改变描述子的取值。
#    改之前请重跑 src/extract_embeddings_128.py 并重新训练模型。

# 检测前把图放大几倍。人脸越小越需要放大，代价是耗时按平方增长。
DETECT_UPSAMPLE = 1

# 描述子采样次数。实测：0 与 1 都是确定性的且结果相同，≥2 才真正抖动。
NUM_JITTERS = 1

# ---------------------------------------------------------------- 异常


class FaceError(RuntimeError):
    """人脸处理失败的基类。"""


class NoFaceDetected(FaceError):
    """图中没有检测到人脸。"""


# ---------------------------------------------------------------- 模型加载


class DlibModels:
    """三个 dlib 模型的打包容器。

    刻意不用 NamedTuple：这里持有的是不可序列化的 C++ 对象，
    保持成一个普通类可以避免有人误以为它能被 pickle 进模型文件。
    """

    def __init__(self, detector, predictor, facerec):
        self.detector = detector
        self.predictor = predictor
        self.facerec = facerec


def load_models(models_dir=DEFAULT_MODELS_DIR, include_facerec=True):
    """加载 dlib 人脸模型；仅提取关键点时可不加载描述子网络。"""
    models_dir = Path(models_dir)
    predictor_path = models_dir / PREDICTOR_FILE
    facerec_path = models_dir / FACEREC_FILE

    required_models = [
        (predictor_path, "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"),
    ]
    if include_facerec:
        required_models.append(
            (facerec_path, "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2")
        )
    for path, url in required_models:
        if not path.exists():
            raise FileNotFoundError(f"未找到模型文件: {path}\n下载: {url}")

    return DlibModels(
        detector=dlib.get_frontal_face_detector(),
        predictor=dlib.shape_predictor(str(predictor_path)),
        facerec=(
            dlib.face_recognition_model_v1(str(facerec_path))
            if include_facerec else None
        ),
    )


# ---------------------------------------------------------------- 单张处理


def read_image(img_path):
    """读成 RGB 的 uint8 数组。

    ⚠️ 不要换成 dlib.load_rgb_image：数据集里有带 alpha 通道的 PNG，
       dlib 读这种图会异常，导致人脸检测直接失败（实测有一张图
       在所有上采样级别都检不出人脸）。
    """
    img_path = Path(img_path)
    if not img_path.exists():
        raise FileNotFoundError(f"图片不存在: {img_path}")
    return np.array(Image.open(img_path).convert("RGB"))


def detect_faces(img, models, upsample=DETECT_UPSAMPLE):
    """返回人脸方框列表（按面积从大到小排序）。

    ⚠️ dlib.get_frontal_face_detector() 返回的框【不带置信度】。
       需要置信度请改用 models.detector.run(img, upsample)，
       它会返回 (rectangles, scores, idx) 三元组。
    """
    rects = list(models.detector(img, upsample))
    return sorted(rects, key=lambda r: r.width() * r.height(), reverse=True)


def _extract_landmark_data(img_path, models, upsample):
    """返回已读图像、关键点模型输入和 68 个原图坐标系关键点。"""
    img = img_path if isinstance(img_path, np.ndarray) else read_image(img_path)

    rects = detect_faces(img, models, upsample)
    if not rects:
        raise NoFaceDetected(f"未检测到人脸（上采样 {upsample} 次）")
    shape = models.predictor(img, rects[0])
    landmarks = np.array(
        [[shape.part(k).x, shape.part(k).y] for k in range(N_LANDMARKS)],
        dtype=float,
    )
    return img, shape, landmarks


def extract_landmarks(img_path, models, upsample=DETECT_UPSAMPLE):
    """单张图片 → 原图坐标系下的 68×2 关键点数组。"""
    _, _, landmarks = _extract_landmark_data(img_path, models, upsample)
    return landmarks


def extract_descriptor(img_path, models, upsample=DETECT_UPSAMPLE, jitters=NUM_JITTERS):
    """单张图片 → (128 维描述子, 68×2 关键点数组)。

    img_path 也可以直接传已读好的 RGB 数组，避免重复读盘。

    多张人脸时取面积最大的那张。检测不到人脸抛 NoFaceDetected。
    """
    if models.facerec is None:
        raise RuntimeError("当前只加载了关键点模型，无法提取 128 维描述子")
    img, shape, landmarks = _extract_landmark_data(img_path, models, upsample)
    desc = models.facerec.compute_face_descriptor(img, shape, jitters)
    return np.asarray(desc, dtype=float), landmarks
