# -*- coding: utf-8 -*-
"""
按论文表 1（35 个距离特征）+ 表 2（26 个角度特征）从 68 关键点重建几何特征。

论文：喻为栋, 徐青云, 刘雷. 小样本条件下利用面部特征提高罕见病预测准确率的方法研究[J].
      中国数字医学, 2024, 19(10): 20-27. DOI: 10.3969/j.issn.1673-7571.2024.10.004

输入：data/landmarks_68.csv        (117 行 × [5 元数据 + 68 点 × (x,y)])
输出：data/geometric_features_61.csv (117 行 × [5 元数据 + 61 几何特征])

归一化（论文 1.2 节）：
    距离特征 = 两点间距离 / d(点0, 点16)
    角度特征 = |水平夹角余弦值| = |Δx| / 线段长度
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SRC = DATA / "landmarks_68.csv"
DST = DATA / "geometric_features_61.csv"

META_COLS = ["image_id", "subject_id", "label", "raw_path", "processed_path"]

N_LANDMARKS = 68
NORMALIZE_PAIR = (0, 16)  # 论文 1.2 节：以点 0 到点 16 的距离为"脸部宽度"

# ==========================================================================
# 表 1：35 个距离特征
#   每项 = (特征名, 点序)。按点序依次连接相邻点求欧氏距离后求和；
#   只给两个点时即为两点间距离。
#   注：论文表 1 注明"表中特征名称中的左右相对图本身而言"。
# ==========================================================================
DIST_FEATURES = [
    ("cheek_width1",          [2, 14]),
    ("cheek_width2",          [4, 12]),
    ("cheek_width3",          [6, 10]),
    ("left_eyebrow_length",   [17, 18, 19, 20, 21]),
    ("right_eyebrow_length",  [22, 23, 24, 25, 26]),
    ("left_eye_width",        [36, 39]),
    ("left_eye_height",       [38, 40]),
    ("right_eye_width",       [42, 45]),
    ("right_eye_height",      [43, 47]),
    ("nose_bridge_length",    [27, 30]),
    ("nose_tip_width",        [31, 32, 33, 34, 35]),
    ("mouth_width",           [48, 50]),
    ("mouth_height",          [51, 57]),
    ("top_lip_length",        [48, 49, 50, 51, 52, 53, 54]),
    ("top_lip_height",        [51, 62]),
    # ⚠️ 论文表 1 第 16 项原文写作 "点64,55,56,57,58,59,60"。
    #    该点序几何上存疑：64/60 是内唇右/左角，55-59 是外唇下缘，
    #    而对称的 top_lip_length 用的是外唇序列 48..54。
    #    此处【按论文原文实现】，另提供对称修正版（54,55,56,57,58,59,48）
    #    供对照，见 BOTTOM_LIP_LENGTH_ALT。
    ("bottom_lip_length",     [64, 55, 56, 57, 58, 59, 60]),
    ("bottom_lip_height",     [66, 57]),
    ("face_height",           [27, 8]),
    ("eyebrows_dist",         [21, 22]),
    ("eyes_dist",             [39, 42]),
    ("cheek2lefteyebrow",     [0, 17]),
    ("cheek2righteyebrow",    [16, 26]),
    ("cheek2lefteye",         [0, 36]),
    ("cheek2righteye",        [16, 45]),
    ("cheek2nosetip1",        [2, 31]),
    ("cheek2nosetip2",        [14, 35]),
    ("cheek2mouth1",          [4, 48]),
    ("cheek2mouth2",          [12, 54]),
    ("cheek2mouth3",          [8, 57]),
    ("nosetip2lefteye",       [33, 39]),
    ("nosetip2righteye",      [33, 42]),
    ("nosetip2toplip",        [33, 51]),
    ("nosetip2bridge",        [33, 30]),
    ("nosetip2mouth1",        [33, 48]),
    ("nosetip2mouth2",        [33, 54]),
]

# 表 1 第 16 项的对称修正版（作为附加列输出，便于对照，不参与主流程）
BOTTOM_LIP_LENGTH_ALT = ("bottom_lip_length_alt", [54, 55, 56, 57, 58, 59, 48])

# ==========================================================================
# 表 2：26 个角度特征
#   每项 = (特征名, (起点, 终点))，取线段水平夹角余弦值的绝对值：
#       |cos θ| = |x_end - x_start| / 线段长度
# ==========================================================================
ANGLE_FEATURES = [
    ("lefteyebrow1",     (19, 21)),
    ("lefteyebrow2",     (17, 19)),
    ("righteyebrow1",    (22, 24)),
    ("righteyebrow2",    (24, 26)),
    ("lefteye",          (36, 39)),
    ("righteye",         (42, 45)),
    ("lefteye1",         (36, 27)),
    ("righteye1",        (27, 42)),
    ("nose_bridge_tip1", (27, 31)),
    ("nose_bridge_tip2", (27, 35)),
    ("nose_left_wing",   (30, 31)),
    ("nose_right_wing",  (30, 35)),
    ("left_nosetip",     (31, 33)),
    ("right_nosetip",    (33, 35)),
    ("left_toplip1",     (48, 50)),
    ("left_toplip2",     (48, 62)),
    ("right_toplip1",    (52, 54)),
    ("right_toplip2",    (54, 62)),
    ("left_bottomlip1",  (60, 57)),
    ("left_bottomlip2",  (60, 66)),
    ("right_bottomlip1", (64, 57)),
    ("right_bottomlip2", (64, 66)),
    ("left_cheek1",      (0, 8)),
    ("right_cheek1",     (16, 8)),
    ("left_cheek2",      (4, 8)),
    ("right_cheek2",     (12, 8)),
]

EPS = 1e-8

FEATURE_NAMES = (
    [name for name, _ in DIST_FEATURES]
    + [name for name, _ in ANGLE_FEATURES]
)

FEATURE_LABELS_CN = {
    "cheek_width1": "脸颊上部宽度（点2–14）",
    "cheek_width2": "脸颊中部宽度（点4–12）",
    "cheek_width3": "脸颊下部宽度（点6–10）",
    "left_eyebrow_length": "左侧眉毛长度（点17–21）",
    "right_eyebrow_length": "右侧眉毛长度（点22–26）",
    "left_eye_width": "左眼宽度（点36–39）",
    "left_eye_height": "左眼高度（点38–40）",
    "right_eye_width": "右眼宽度（点42–45）",
    "right_eye_height": "右眼高度（点43–47）",
    "nose_bridge_length": "鼻梁长度（点27–30）",
    "nose_tip_width": "鼻尖宽度（点31–35）",
    "mouth_width": "嘴部宽度（点48–50）",
    "mouth_height": "嘴部高度（点51–57）",
    "top_lip_length": "上唇长度（点48–54）",
    "top_lip_height": "上唇高度（点51–62）",
    "bottom_lip_length": "下唇长度（论文定义点64、55–60）",
    "bottom_lip_height": "下唇高度（点66–57）",
    "face_height": "面部高度（点27–8）",
    "eyebrows_dist": "两眉间距（点21–22）",
    "eyes_dist": "两眼间距（点39–42）",
    "cheek2lefteyebrow": "左侧脸颊至眉毛距离（点0–17）",
    "cheek2righteyebrow": "右侧脸颊至眉毛距离（点16–26）",
    "cheek2lefteye": "左侧脸颊至眼睛距离（点0–36）",
    "cheek2righteye": "右侧脸颊至眼睛距离（点16–45）",
    "cheek2nosetip1": "左侧脸颊至鼻部距离（点2–31）",
    "cheek2nosetip2": "右侧脸颊至鼻部距离（点14–35）",
    "cheek2mouth1": "左侧脸颊至嘴部距离（点4–48）",
    "cheek2mouth2": "右侧脸颊至嘴部距离（点12–54）",
    "cheek2mouth3": "下巴至嘴部距离（点8–57）",
    "nosetip2lefteye": "鼻尖至左眼距离（点33–39）",
    "nosetip2righteye": "鼻尖至右眼距离（点33–42）",
    "nosetip2toplip": "鼻尖至上唇距离（点33–51）",
    "nosetip2bridge": "鼻尖至鼻梁距离（点33–30）",
    "nosetip2mouth1": "鼻尖至左侧嘴角距离（点33–48）",
    "nosetip2mouth2": "鼻尖至右侧嘴角距离（点33–54）",
    "lefteyebrow1": "左眉倾斜度（点19–21）",
    "lefteyebrow2": "左眉倾斜度（点17–19）",
    "righteyebrow1": "右眉倾斜度（点22–24）",
    "righteyebrow2": "右眉倾斜度（点24–26）",
    "lefteye": "左眼倾斜度（点36–39）",
    "righteye": "右眼倾斜度（点42–45）",
    "lefteye1": "左眼至鼻梁方向（点36–27）",
    "righteye1": "鼻梁至右眼方向（点27–42）",
    "nose_bridge_tip1": "鼻梁至鼻翼方向（点27–31）",
    "nose_bridge_tip2": "鼻梁至鼻翼方向（点27–35）",
    "nose_left_wing": "左鼻翼倾斜度（点30–31）",
    "nose_right_wing": "右鼻翼倾斜度（点30–35）",
    "left_nosetip": "左侧鼻尖倾斜度（点31–33）",
    "right_nosetip": "右侧鼻尖倾斜度（点33–35）",
    "left_toplip1": "左侧上唇倾斜度（点48–50）",
    "left_toplip2": "左侧上唇方向（点48–62）",
    "right_toplip1": "右侧上唇倾斜度（点52–54）",
    "right_toplip2": "右侧上唇方向（点54–62）",
    "left_bottomlip1": "左侧下唇倾斜度（点60–57）",
    "left_bottomlip2": "左侧下唇方向（点60–66）",
    "right_bottomlip1": "右侧下唇倾斜度（点64–57）",
    "right_bottomlip2": "右侧下唇方向（点64–66）",
    "left_cheek1": "左侧脸部轮廓方向（点0–8）",
    "right_cheek1": "右侧脸部轮廓方向（点16–8）",
    "left_cheek2": "左侧脸颊方向（点4–8）",
    "right_cheek2": "右侧脸颊方向（点12–8）",
}


# ==========================================================================
# 几何计算
# ==========================================================================
def polyline_length(pts, idxs):
    """按给定点序依次连接相邻点，求欧氏距离之和。"""
    p = pts[idxs]
    segs = np.linalg.norm(np.diff(p, axis=0), axis=1)
    return float(segs.sum())


def horizontal_angle_cosine(pts, pair):
    """线段水平夹角余弦值的绝对值 |Δx| / L。"""
    a, b = pair
    dx = pts[b][0] - pts[a][0]
    length = float(np.linalg.norm(pts[b] - pts[a]))
    return abs(dx) / (length + EPS)


def extract_landmark_features(landmarks, include_alt=False):
    """从原图坐标系下的 68 个关键点提取几何特征。"""
    pts = np.asarray(landmarks, dtype=float)
    if pts.shape != (N_LANDMARKS, 2):
        raise RuntimeError(
            f"关键点形状为 {pts.shape}，应为 {(N_LANDMARKS, 2)}"
        )
    if not np.isfinite(pts).all():
        raise RuntimeError("关键点中存在 NaN 或无穷值")

    a, b = NORMALIZE_PAIR
    scale = float(np.linalg.norm(pts[a] - pts[b]))
    if scale < EPS:
        raise RuntimeError("脸部宽度为 0，无法归一化几何特征")

    out = {
        name: polyline_length(pts, idxs) / scale
        for name, idxs in DIST_FEATURES
    }
    if include_alt:
        name, idxs = BOTTOM_LIP_LENGTH_ALT
        out[name] = polyline_length(pts, idxs) / scale
    out.update({
        name: horizontal_angle_cosine(pts, pair)
        for name, pair in ANGLE_FEATURES
    })
    return out


def feature_vector_from_landmarks(landmarks):
    """返回按 FEATURE_NAMES 排列的 61 维几何特征向量。"""
    values = extract_landmark_features(landmarks)
    return np.asarray([values[name] for name in FEATURE_NAMES], dtype=float)


# ==========================================================================
# 主流程
# ==========================================================================
def extract(df):
    coords = df.drop(columns=META_COLS).to_numpy(dtype=float)
    n = len(df)
    if coords.shape[1] != N_LANDMARKS * 2:
        raise RuntimeError(
            f"坐标列数为 {coords.shape[1]}，应为 {N_LANDMARKS * 2}"
        )

    landmarks = coords.reshape(n, N_LANDMARKS, 2)

    # 归一化基准：点 0 到点 16 的距离（论文称"脸部宽度"）
    scale = np.linalg.norm(landmarks[:, 0] - landmarks[:, 16], axis=1)
    if (scale < EPS).any():
        bad = df.loc[scale < EPS, "image_id"].tolist()
        raise RuntimeError(f"存在脸部宽度为 0 的样本: {bad}")

    rows = [
        extract_landmark_features(points, include_alt=True)
        for points in landmarks
    ]
    return pd.DataFrame(rows, index=df.index)


def main():
    if not SRC.exists():
        raise FileNotFoundError(f"未找到输入文件: {SRC}")

    df = pd.read_csv(SRC, encoding="utf-8-sig")
    print(f"输入: {SRC.name}  →  {df.shape[0]} 行 × {df.shape[1]} 列")

    feats = extract(df)

    n_dist = len(DIST_FEATURES)
    n_angle = len(ANGLE_FEATURES)
    print(f"生成: {n_dist} 距离特征 + {n_angle} 角度特征 = {n_dist + n_angle} 维"
          f"（另含 1 个附加对照列）")

    result = pd.concat([df[META_COLS], feats], axis=1)
    result.to_csv(DST, index=False, encoding="utf-8-sig")
    print(f"输出: {DST.name}  →  {result.shape[0]} 行 × {result.shape[1]} 列")

    # ---- 质量核查（论文第 2 页 1.2 节称特征值"符合正态分布，未出现异常值"）----
    print("\n=== 特征质量核查 ===")
    vals = feats.to_numpy()
    print(f"缺失值      : {int(np.isnan(vals).sum())}")
    print(f"全局范围    : {vals.min():.4f} ~ {vals.max():.4f}")

    dist_cols = [n for n, _ in DIST_FEATURES]
    angle_cols = [n for n, _ in ANGLE_FEATURES]
    d = feats[dist_cols].to_numpy()
    a_ = feats[angle_cols].to_numpy()
    print(f"距离特征范围: {d.min():.4f} ~ {d.max():.4f}  (归一化后应在 0~2 量级)")
    print(f"角度特征范围: {a_.min():.4f} ~ {a_.max():.4f}  (|cos| 应落在 0~1)")

    ok = (d > 0).all() and (a_ >= 0).all() and (a_ <= 1).all()
    print(f"取值合法性  : {'✅ 通过' if ok else '❌ 存在异常'}")

    return result


def report_feature_importance(df, top_n=10):
    """用随机森林评估 61 维特征重要性，对应论文表 3 / 图 6。"""
    from sklearn.ensemble import RandomForestClassifier

    lab = {"digeorge": 0, "down": 1, "williams": 2}
    y = df["label"].map(lab).to_numpy()
    X = df.drop(columns=META_COLS).to_numpy(dtype=float)

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    names = [c for c in df.columns if c not in META_COLS]
    imp = pd.Series(rf.feature_importances_, index=names).sort_values(ascending=False)
    return imp, imp.head(top_n)


if __name__ == "__main__":
    main()
