# -*- coding: utf-8 -*-
"""
论文步骤 6：特征融合 —— 10 维几何特征 + 128 维深度特征 = 138 维。

论文原文（第 4 页）：
    "通过将几何特征与 Dlib 深度学习特征融合，特征维度从 128 维变成 138 维，
     测试表明这进一步提升了分类效果。"

融合方式：直接拼接（concatenation）
    X = [ geom_top10 (10维) , dlib_embedding (128维) ]  → 138 维

输入：data/geometric_features_61.csv  (61 维几何特征)
      data/embeddings_128.csv         (128 维深度特征)
输出：data/features_138.csv           (meta + 10 几何 + 128 深度)
      results/geo61/feature_importance_61.csv (61 维特征重要性)

⚠️ 关于几何特征的选择：
   论文 2.2 节用随机森林选 Top10。此处按【全量数据】做一次选择并固定，
   这与论文的描述方式一致。另在训练脚本中提供了【折内选择】的无泄漏版本，
   两者实测差异极小（见 step.md）。
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
GEO_SRC = DATA / "geometric_features_61.csv"
EMB_SRC = DATA / "embeddings_128.csv"
DST = DATA / "features_138.csv"

META_COLS = ["image_id", "subject_id", "label", "raw_path", "processed_path"]
LABEL_MAP = {"digeorge": 0, "down": 1, "williams": 2}

N_SELECT = 10          # 论文 2.2 节：保留前 10 个最重要特征
SEED = 42
EXCLUDE = {"bottom_lip_length_alt"}   # 非论文定义的附加对照列


def main():
    geo = pd.read_csv(GEO_SRC, encoding="utf-8-sig")
    emb = pd.read_csv(EMB_SRC, encoding="utf-8-sig")

    geom_cols = [
        c for c in geo.columns if c not in META_COLS and c not in EXCLUDE
    ]
    emb_cols = [c for c in emb.columns if c.startswith("e") and c[1:].isdigit()]

    print(f"几何特征: {len(geom_cols)} 维  ← {GEO_SRC.name}")
    print(f"深度特征: {len(emb_cols)} 维  ← {EMB_SRC.name}")

    # ---- 按 image_id 对齐，确保顺序一致 ----
    merged = geo[META_COLS + geom_cols].merge(
        emb[META_COLS[:1] + emb_cols], on="image_id", how="inner", validate="1:1"
    )
    if len(merged) != len(geo) or len(merged) != len(emb):
        raise RuntimeError(
            f"对齐后行数不一致: geo={len(geo)}, emb={len(emb)}, merged={len(merged)}"
        )
    print(f"对齐后样本数: {len(merged)}")

    # ---- 随机森林选 Top10 几何特征 ----
    y = merged["label"].map(LABEL_MAP).to_numpy()
    if np.isnan(y).any():
        raise RuntimeError("存在无法映射的标签")

    X_geo = merged[geom_cols].to_numpy(dtype=float)
    rf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    rf.fit(X_geo, y)

    imp = pd.Series(rf.feature_importances_, index=geom_cols).sort_values(
        ascending=False
    )
    top10 = list(imp.head(N_SELECT).index)

    print(f"\n随机森林选出的 Top{N_SELECT} 几何特征:")
    for i, name in enumerate(top10, 1):
        print(f"  {i:2d}. {name:22s} {imp[name]:.6f}")

    # ---- 拼接 ----
    out = pd.concat(
        [merged[META_COLS + top10 + emb_cols]], axis=1
    )
    out.to_csv(DST, index=False, encoding="utf-8-sig")

    n_total = len(top10) + len(emb_cols)
    print(f"\n输出: {DST.name}")
    print(f"  {len(top10)} 几何 + {len(emb_cols)} 深度 = {n_total} 维"
          f"  （目标 138 维: {'✅' if n_total == 138 else '❌'}）")
    print(f"  样本数 {len(out)}，总列数 {out.shape[1]}")

    # 记录选中的几何特征，供训练脚本复现
    out_dir = ROOT / "results" / "geo61"
    out_dir.mkdir(parents=True, exist_ok=True)
    imp.to_csv(out_dir / "feature_importance_61.csv",
               encoding="utf-8-sig", header=["importance"])
    print(f"  特征重要性已存: results/geo61/feature_importance_61.csv")

    return out


if __name__ == "__main__":
    main()
