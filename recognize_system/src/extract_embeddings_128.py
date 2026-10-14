# -*- coding: utf-8 -*-
"""
论文步骤 5：用 Dlib 提取 128 维深度特征。

论文原文（1.2 节 / 2.3 节）：
    "选用 Dlib 的 128 维向量作为面部总体特征集，它是由一个基于 ResNet 的深度学习网络生成，
     经过修改以适应面部识别任务。Dlib 通过增加全局平均池化层，将卷积层的输出转换成
     128 维的特征向量，这一向量就是面部的深度学习特征。"

对应模型文件：dlib_face_recognition_resnet_model_v1.dat

输入：data/raw/<label>/<subject>/<image>.jpg
输出：data/embeddings_128.csv      (117 行 × [5 元数据 + 128 维向量])
      data/landmarks_68_raw.csv   (在原图上新检测的关键点，供追溯)

⚠️ 为什么不用已有的 landmarks_68.csv？
   经核对，该文件的关键点位于【裁剪后人脸图】(约 256×256) 的坐标系，
   而本项目没有裁剪图（data/processed/faces/ 不存在）。
   把它套到 raw 原图上会导致描述子完全失效 —— 实测异人图片对的
   平均距离仅 0.425，94.9% 低于 dlib 判定阈值 0.6，向量已丧失区分力。

   注：这【不影响】geometric_features_61.csv —— 几何特征做了
   ÷d(0,16) 与 |Δx|/L 归一化，是尺度与平移无关的，坐标系不同结果不变。

⚠️ 关键点必须传入 compute_face_descriptor —— dlib 内部会用它们做 5 点仿射对齐，
   若只用检测框，128 维向量的稳定性会显著下降。
"""

from pathlib import Path
import numpy as np
import pandas as pd

from face_embedding import (
    DETECT_UPSAMPLE,
    N_EMBEDDING,
    N_LANDMARKS,
    NUM_JITTERS,
    extract_descriptor,
    load_models,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LANDMARKS = DATA / "landmarks_68.csv"
DST = DATA / "embeddings_128.csv"
DST_LANDMARKS = DATA / "landmarks_68_raw.csv"

META_COLS = ["image_id", "subject_id", "label", "raw_path", "processed_path"]


def resolve_path(raw_path: str) -> Path:
    """CSV 中的 raw_path 形如 data/raw/<label>/<subject>/xx.jpg，直接映射到项目根下。"""
    return ROOT / raw_path


def extract(df: pd.DataFrame, models):
    """在 raw 原图上检测人脸 → 68 关键点 → 128 维描述子。

    返回 (描述子矩阵, 原图坐标系下的关键点数组, 失败列表)。

    核心逻辑在 face_embedding.extract_descriptor —— 与推理脚本共用同一份代码，
    保证训练和推理算出来的是同一种特征。
    """
    vectors, landmark_rows, failures = [], [], []

    for i, row in df.iterrows():
        try:
            desc, xy = extract_descriptor(resolve_path(row["raw_path"]), models)
            vectors.append(desc)
            landmark_rows.append(xy)
        except Exception as exc:  # noqa: BLE001
            failures.append((row["image_id"], f"{type(exc).__name__}: {exc}"))
            vectors.append(np.full(N_EMBEDDING, np.nan))
            landmark_rows.append(np.full((N_LANDMARKS, 2), np.nan))

        if (i + 1) % 20 == 0:
            print(f"  已处理 {i + 1}/{len(df)}")

    return np.vstack(vectors), np.stack(landmark_rows), failures


def sanity_check(emb: np.ndarray, df: pd.DataFrame):
    """验证描述子质量：同一受试者的不同照片应比不同受试者更接近。

    这是判断"关键点是否与原图对齐"的关键校验 —— 若对齐错误，描述子会失效，
    同人距离就不会明显小于异人距离。
    """
    valid = ~np.isnan(emb).any(axis=1)
    e, d = emb[valid], df[valid].reset_index(drop=True)

    # 注：dlib 描述子【不是】单位向量（实测 L2 范数均值 ≈ 1.37，范围 1.24~1.50），
    #     所以欧氏距离并不等于 sqrt(2-2cos)，dlib 官方直接按欧氏距离比较。
    same, diff = [], []
    subj = d["subject_id"].to_numpy()
    for i in range(len(e)):
        for j in range(i + 1, len(e)):
            dist = float(np.linalg.norm(e[i] - e[j]))
            (same if subj[i] == subj[j] else diff).append(dist)

    same, diff = np.array(same), np.array(diff)
    print()
    print("=== 描述子质量校验（关键点对齐验证）===")
    print(f"同一受试者图片对: {len(same):5d} 对   距离 均值 {same.mean():.3f}  中位 {np.median(same):.3f}")
    print(f"不同受试者图片对: {len(diff):5d} 对   距离 均值 {diff.mean():.3f}  中位 {np.median(diff):.3f}")
    print(f"区分度 (异人均值 - 同人均值): {diff.mean() - same.mean():.3f}")
    print(f"以 dlib 官方阈值 0.6 判定同一人:")
    print(f"  同人 命中率(应高): {(same < 0.6).mean() * 100:.1f}%")
    print(f"  异人 误判率(应低): {(diff < 0.6).mean() * 100:.1f}%")
    print()
    if same.mean() < diff.mean() - 0.2:
        print("  ✅ 同人距离显著小于异人距离 —— 关键点与原图对齐正确")
    else:
        print("  ❌ 区分度不足 —— 关键点可能未与原图对齐")


def main():
    # 只取元数据列，关键点从原图重新检测
    df = pd.read_csv(LANDMARKS, encoding="utf-8-sig")[META_COLS]
    print(f"输入: data/raw/  →  {len(df)} 张")
    print(f"检测上采样  : {DETECT_UPSAMPLE}   描述子采样: {NUM_JITTERS}\n")

    models = load_models()

    emb, landmarks, failures = extract(df, models)

    if failures:
        print(f"\n⚠️ 失败 {len(failures)} 张:")
        for iid, reason in failures:
            print(f"   {iid}: {reason}")
    else:
        print(f"\n全部 {len(df)} 张提取成功")

    # 保存原图坐标系下的关键点
    lm_cols = [f"{ax}{k}" for k in range(N_LANDMARKS) for ax in ("x", "y")]
    pd.concat(
        [df[META_COLS].reset_index(drop=True),
         pd.DataFrame(landmarks.reshape(len(df), -1), columns=lm_cols)],
        axis=1,
    ).to_csv(DST_LANDMARKS, index=False, encoding="utf-8-sig")
    print(f"输出: {DST_LANDMARKS.name}  (原图坐标系关键点)")

    cols = [f"e{i}" for i in range(N_EMBEDDING)]
    out = pd.concat(
        [df[META_COLS].reset_index(drop=True), pd.DataFrame(emb, columns=cols)],
        axis=1,
    )
    out.to_csv(DST, index=False, encoding="utf-8-sig")
    print(f"输出: {DST.name}  →  {out.shape[0]} 行 × {out.shape[1]} 列")

    sanity_check(emb, df)
    return out


if __name__ == "__main__":
    main()
