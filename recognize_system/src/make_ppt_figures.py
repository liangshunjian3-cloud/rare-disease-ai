# -*- coding: utf-8 -*-
"""
生成中期报告 PPT 所需的全部图表素材 → docs/ppt_assets/

配色与 docs/MIDTERM_PPT.md 的全局视觉规范一致，保证插图与幻灯片同色系。

运行：
    PYTHONIOENCODING=utf-8 python src/make_ppt_figures.py
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results" / "fusion138"
OUT = ROOT / "docs" / "ppt_assets"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 配色
NAVY = "#1F3864"     # 主色
BLUE = "#2E75B6"     # 强调
LBLUE = "#9DC3E6"    # 浅蓝
RED = "#C00000"      # 警示
GREEN = "#548235"    # 通过
GRAY = "#7F7F7F"     # 次要
DGRAY = "#404040"    # 正文
ORANGE = "#ED7D31"

PALETTE = [NAVY, BLUE, LBLUE, GRAY]

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei"],
    "axes.unicode_minus": False,
    "font.size": 13,
    "axes.edgecolor": "#BFBFBF",
    "axes.labelcolor": DGRAY,
    "text.color": DGRAY,
    "xtick.color": DGRAY,
    "ytick.color": DGRAY,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.transparent": True,
})


def save(fig, name, pad=0.05):
    path = OUT / name
    fig.savefig(path, transparent=True, pad_inches=pad)
    plt.close(fig)
    print(f"  ✓ {name}")


def clean(ax, grid_axis="y"):
    """去掉上/右边框，加淡网格。"""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color="#E7E6E6", lw=0.8, zorder=0)
    ax.set_axisbelow(True)


# ======================================================================
# P7 数据集
# ======================================================================
def fig_dataset_bar():
    """各病种训练样本数与受试者数。"""
    labels = ["唐氏综合征\nDown", "Williams\n综合征", "DiGeorge\n综合征"]
    imgs = [53, 33, 31]
    subjs = [50, 33, 26]

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    x = np.arange(3)
    w = 0.38
    b1 = ax.bar(x - w / 2, imgs, w, label="训练图片数", color=NAVY, zorder=3)
    b2 = ax.bar(x + w / 2, subjs, w, label="受试者数", color=LBLUE, zorder=3)

    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.9,
                    f"{int(b.get_height())}", ha="center", va="bottom",
                    fontsize=13, fontweight="bold", color=DGRAY)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("数量", fontsize=12)
    ax.set_ylim(0, 62)
    clean(ax)
    ax.legend(frameon=False, fontsize=11, loc="upper right")
    ax.set_title("数据集构成：3 类罕见遗传综合征", fontsize=14,
                 fontweight="bold", color=NAVY, pad=12)
    save(fig, "dataset_bar.png")


def fig_detect_funnel():
    """330 → 检出 127 → 训练 117 的漏斗。"""
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    stages = ["原始图片池", "可检出人脸", "进入训练集"]
    vals = [330, 127, 117]
    colors = [GRAY, BLUE, NAVY]
    y = np.arange(3)[::-1]

    ax.barh(y, vals, height=0.5, color=colors, zorder=3)
    for yi, v, s in zip(y, vals, stages):
        ax.text(v + 9, yi, f"{v}", va="center", fontsize=16,
                fontweight="bold", color=DGRAY)
        ax.text(9, yi, s, va="center", fontsize=12.5, color="white",
                fontweight="bold", zorder=4)

    # 流失原因直接写在条下方，避免箭头与色条打架
    ax.text(127 + 9, y[1] - 0.34, "↓  漏检 203 张（62%）：侧脸 / 仰角 / 眼部特写",
            va="center", fontsize=10.5, color=RED)
    ax.text(117 + 9, y[2] - 0.34, "↓  数据集未选用 10 张",
            va="center", fontsize=10.5, color=GRAY)

    ax.set_xlim(0, 430)
    ax.set_ylim(-0.78, 2.55)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#E7E6E6", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("从原始图片到训练集：整体检出率 38%", fontsize=14,
                 fontweight="bold", color=NAVY, pad=12)
    save(fig, "detect_funnel.png")


def fig_detect_rate():
    """按文件名的检出率——正面照约束的证据。"""
    stems = ["01", "02", "03", "04", "05", "06~08"]
    total = [113, 95, 60, 42, 14, 6]
    det = [110, 12, 2, 2, 1, 0]
    rate = [d / t * 100 for d, t in zip(det, total)]

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    colors = [GREEN] + [RED] * 5
    bars = ax.bar(stems, rate, color=colors, width=0.6, zorder=3)
    for b, r, d, t in zip(bars, rate, det, total):
        ax.text(b.get_x() + b.get_width() / 2, r + 2.5, f"{r:.0f}%",
                ha="center", fontsize=13, fontweight="bold",
                color=DGRAY)
        ax.text(b.get_x() + b.get_width() / 2, -7.5, f"{d}/{t}",
                ha="center", fontsize=10, color=GRAY)

    ax.axhline(0, color="#BFBFBF", lw=1)
    ax.set_ylim(-12, 118)
    ax.set_ylabel("人脸检出率 (%)", fontsize=12)
    ax.set_xlabel("文件名序号（数据集约定 01 为正面照）", fontsize=12)
    clean(ax)
    ax.set_title("正面照约束：01 检出 97%，其余仅 3~13%", fontsize=14,
                 fontweight="bold", color=NAVY, pad=12)
    save(fig, "detect_rate.png")


# ======================================================================
# P8 关键点
# ======================================================================
def fig_landmarks():
    """真实样本上的 68 关键点 + 5 点对齐示意。"""
    lm = pd.read_csv(DATA / "landmarks_68_raw.csv", encoding="utf-8-sig")
    # 挑一张正面照
    row = lm[lm["raw_path"].str.contains("digeorge-nigeria-patient-1/01")].iloc[0]
    img_path = ROOT / row["raw_path"]
    img = np.array(Image.open(img_path).convert("RGB"))

    pts = np.array([[row[f"x{k}"], row[f"y{k}"]] for k in range(68)])

    # dlib 68 点分组（用于 5 点仿射对齐的是 36,45,30,48,54 五个点）
    GROUPS = {
        "下颌": (0, 17), "右眉": (17, 22), "左眉": (22, 27),
        "鼻梁": (27, 31), "鼻底": (31, 36),
        "右眼": (36, 42), "左眼": (42, 48),
        "上唇外": (48, 55), "下唇外": (55, 60),
        "上唇内": (60, 65), "下唇内": (65, 68),
    }
    ALIGN5 = {30: "鼻尖", 36: "右眼角", 45: "左眼角", 48: "右嘴角", 54: "左嘴角"}

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6))

    # ---- 左：全部 68 点分区着色 ----
    ax = axes[0]
    ax.imshow(img)
    cmap = plt.get_cmap("tab20")
    for i, (name, (a, b)) in enumerate(GROUPS.items()):
        ax.plot(pts[a:b, 0], pts[a:b, 1], "o-", ms=2.6, lw=1.2,
                color=cmap(i / 20), alpha=0.9)
    for k in ALIGN5:
        ax.plot(pts[k, 0], pts[k, 1], "*", ms=17, color="#FFFF00",
                markeredgecolor=RED, markeredgewidth=1.3, zorder=5)
    ax.set_title("68 个关键点（分区着色）", fontsize=13,
                 fontweight="bold", color=NAVY, pad=8)
    ax.axis("off")

    # ---- 右：5 点仿射对齐 ----
    ax = axes[1]
    ax.imshow(img)
    ax.plot(pts[:, 0], pts[:, 1], "o", ms=1.8, color=LBLUE, alpha=0.8)
    tri = [30, 36, 45, 48, 54]
    # 每个点的标签偏移与对齐方式，避免标签互相压盖
    OFFSETS = {
        36: ((-42, 2), "right"),    # 右眼外眼角（图像左侧）
        45: ((42, 2), "left"),      # 左眼外眼角（图像右侧）
        30: ((0, -26), "center"),   # 鼻尖
        48: ((-42, -12), "right"),  # 右嘴角（图像左侧）
        54: ((42, -12), "left"),    # 左嘴角（图像右侧）
    }
    ax.plot(pts[tri, 0], pts[tri, 1], "o", ms=11,
            mfc="none", mec=RED, mew=2.2, zorder=5)
    for k in tri:
        x, y = pts[k]
        off, ha = OFFSETS[k]
        ax.annotate(ALIGN5[k], (x, y), textcoords="offset points",
                    xytext=off, ha=ha, va="center", fontsize=10.5,
                    color=RED, fontweight="bold", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.22", fc="white",
                              ec=RED, lw=0.9, alpha=0.95))
    ax.set_title("5 点仿射对齐基准（送进描述子网络）", fontsize=13,
                 fontweight="bold", color=NAVY, pad=8)
    ax.axis("off")

    fig.suptitle("Dlib 68 点关键点定位（真实样本）", fontsize=15,
                 fontweight="bold", color=NAVY, y=1.02)
    save(fig, "landmarks_face.png")


def fig_geo_dist():
    """61 维几何特征的构成与归一化。"""
    fig, ax = plt.subplots(figsize=(6.6, 3.4))

    parts = [
        ("距离特征", 35, NAVY, "鼻尖–左/右眼、鼻梁长、唇宽 …"),
        ("角度特征", 26, BLUE, r"由关键点构成，取 $|\cos\theta|$"),
    ]
    left = 0
    for name, n, color, desc in parts:
        ax.barh(0.5, n, left=left, height=0.42, color=color, zorder=3)
        ax.text(left + n / 2, 0.5, f"{name}\n{n} 维", ha="center", va="center",
                color="white", fontsize=13, fontweight="bold", zorder=4)
        ax.text(left + n / 2, 0.22, desc, ha="center", va="top",
                fontsize=10, color=GRAY)
        left += n

    ax.text(61 + 1.5, 0.5, "= 61 维", va="center", fontsize=14,
            fontweight="bold", color=NAVY)

    ax.text(0, 0.86, "归一化：距离 ÷ d(0,16)（两眼外眼角距） · 横坐标差 ÷ 人脸宽 L · 角度取 |cos θ|",
            fontsize=10.5, color=GREEN, va="bottom")
    ax.text(0, 0.08, "→ 与图像分辨率、拍摄距离无关，同一张脸在任何尺寸下算出的几何特征一致",
            fontsize=10.5, color=GREEN, va="top")

    ax.set_xlim(-1, 74)
    ax.set_ylim(-0.05, 1.0)
    ax.axis("off")
    ax.set_title("61 维几何特征构成", fontsize=14, fontweight="bold",
                 color=NAVY, pad=14)
    save(fig, "geo_composition.png")


# ======================================================================
# P9 深度特征
# ======================================================================
def fig_desc_distance():
    """同人 vs 异人描述子距离分布 —— 对齐正确性的证据。"""
    emb = pd.read_csv(DATA / "embeddings_128.csv", encoding="utf-8-sig")
    cols = [c for c in emb.columns if c.startswith("e") and c[1:].isdigit()]
    X = emb[cols].to_numpy(float)
    subj = emb["subject_id"].to_numpy()

    same, diff = [], []
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            d = float(np.linalg.norm(X[i] - X[j]))
            (same if subj[i] == subj[j] else diff).append(d)
    same, diff = np.array(same), np.array(diff)

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    bins = np.linspace(0.15, 1.15, 46)
    ax.hist(diff, bins=bins, color=LBLUE, alpha=0.85, zorder=3,
            label=f"不同受试者（{len(diff)} 对，均值 {diff.mean():.3f}）")
    ax.hist(same, bins=bins, color=RED, alpha=0.95, zorder=4,
            label=f"同一受试者（{len(same)} 对，均值 {same.mean():.3f}）")

    ymax = ax.get_ylim()[1]
    ax.axvline(0.6, color=DGRAY, ls="--", lw=1.6, zorder=5)
    ax.text(0.585, ymax * 0.66, "dlib 官方判定阈值 0.6", ha="right",
            fontsize=10.5, color=DGRAY)

    ax.annotate("", xy=(same.mean(), 0), xytext=(diff.mean(), 0),
                arrowprops=dict(arrowstyle="<->", color=GREEN, lw=2.2))
    ax.text((same.mean() + diff.mean()) / 2, -ymax * 0.115,
            f"区分度 {diff.mean() - same.mean():.3f}",
            ha="center", va="center", fontsize=12, fontweight="bold",
            color=GREEN)

    ax.set_xlabel("128 维描述子的欧氏距离", fontsize=12)
    ax.set_ylabel("图片对数量", fontsize=12)
    # 底部留白给「区分度」标注，但刻度只显示非负值
    ax.set_ylim(-ymax * 0.16, ymax * 1.10)
    ax.set_yticks(np.arange(0, ymax * 1.05, 100))
    clean(ax)
    ax.legend(frameon=False, fontsize=10.5, loc="upper left")
    ax.set_title("描述子质量校验：同人 0.350 vs 异人 0.641", fontsize=14,
                 fontweight="bold", color=NAVY, pad=12)
    save(fig, "desc_distance.png")


# ======================================================================
# P10 特征重要性
# ======================================================================
def fig_importance():
    """Top-10 特征重要性（命中论文的用深色）。"""
    feats = [
        ("nosetip2righteye", 0.045429, 3),
        ("righteye", 0.041580, 8),
        ("nosetip2lefteye", 0.038571, 1),
        ("righteyebrow1", 0.034044, 4),
        ("nose_bridge_length", 0.033656, 2),
        ("nose_bridge_tip1", 0.028402, None),
        ("nose_bridge_tip2", 0.025824, None),
        ("bottom_lip_length", 0.023779, None),
        ("nosetip2mouth2", 0.023099, None),
        ("cheek2lefteye", 0.021486, None),
    ]
    names = [f[0] for f in feats][::-1]
    imps = [f[1] for f in feats][::-1]
    paper = [f[2] for f in feats][::-1]
    colors = [NAVY if p else LBLUE for p in paper]

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    y = np.arange(len(names))
    ax.barh(y, imps, color=colors, height=0.66, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=11)

    for yi, imp, p in zip(y, imps, paper):
        ax.text(imp + 0.0008, yi, f"{imp:.4f}", va="center", fontsize=10,
                color=DGRAY)
        if p:
            ax.text(0.0012, yi, f"论文 No.{p}", va="center", fontsize=10,
                    color="white", fontweight="bold", zorder=5)

    ax.set_xlim(0, 0.054)
    ax.set_xlabel("随机森林特征重要性", fontsize=12)
    clean(ax, grid_axis="x")
    ax.set_title("几何特征重要性 Top-10（深色 = 命中论文表 3）",
                 fontsize=14, fontweight="bold", color=NAVY, pad=12)
    save(fig, "feature_importance.png")


def fig_fusion_138():
    """138 维融合特征构成示意。"""
    fig, ax = plt.subplots(figsize=(7.2, 2.5))

    blocks = [
        (0, 10, BLUE, "几何 Top-10", "论文表 3 筛选"),
        (10, 128, NAVY, "Dlib 深度特征 128 维", "ResNet + 全局平均池化"),
    ]
    for left, w, color, title, sub in blocks:
        ax.add_patch(FancyBboxPatch((left, 0.34), w, 0.34,
                                    boxstyle="round,pad=0.02",
                                    fc=color, ec="none", zorder=3))
        ax.text(left + w / 2, 0.51, title, ha="center", va="center",
                color="white", fontsize=12, fontweight="bold", zorder=4)
        ax.text(left + w / 2, 0.24, sub, ha="center", va="top",
                fontsize=10, color=GRAY)

    ax.text(69, 0.51, "= 138 维融合特征", ha="center", va="center",
            fontsize=13, fontweight="bold", color=GREEN)

    ax.set_xlim(-3, 145)
    ax.set_ylim(0.0, 0.85)
    ax.axis("off")
    ax.set_title("融合特征构建", fontsize=14, fontweight="bold",
                 color=NAVY, pad=10)
    save(fig, "fusion_138.png")


# ======================================================================
# P12 三方案对照
# ======================================================================
def read_metrics():
    d = pd.read_csv(RESULTS / "metrics_summary.csv", encoding="utf-8-sig")
    d["auc"] = d["AUC平均值"].str.split("±").str[0].astype(float)
    return d


def fig_auc_comparison():
    """4 模型 × 3 特征方案，叠加论文数值。"""
    d = read_metrics()
    models = ["随机森林", "支持向量机", "XGBoost", "多层感知器"]
    schemes = ["几何特征", "深度学习特征", "融合特征"]
    colors = [LBLUE, BLUE, NAVY]

    # 论文表 4
    PAPER = {
        "随机森林":   [0.79, 0.92, 0.94],
        "支持向量机": [0.79, 0.96, 0.96],
        "XGBoost":    [0.81, 0.94, 0.95],
        "多层感知器": [0.84, 0.96, 0.98],
    }

    fig, ax = plt.subplots(figsize=(9.8, 4.6))
    x = np.arange(len(models))
    w = 0.25

    for si, (sch, c) in enumerate(zip(schemes, colors)):
        vals = [d[(d["建模方法"] == m) & (d["特征方案"] == sch)]["auc"].iloc[0]
                for m in models]
        pos = x + (si - 1) * w
        bars = ax.bar(pos, vals, w, label=sch, color=c, zorder=3)
        # 数值统一写在柱内底部：柱顶要留给论文数值的 ▽ 标记，避免压字
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, 0.708, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9.5, zorder=5,
                    color="white" if si else DGRAY)

        # 论文数值：同位置叠一个红色空心倒三角，直接可比
        paper_vals = [PAPER[m][si] for m in models]
        ax.plot(pos, paper_vals, "v", ms=9, mfc="white", mec=RED, mew=2,
                zorder=6, label="论文表 4 数值" if si == 0 else None)

    # 每个模型上方的分组底色，帮助区分
    for xi in x:
        ax.axvspan(xi - 0.42, xi + 0.42, color="#F2F5FA", zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12.5)
    ax.set_ylabel("AUC（10 折平均）", fontsize=12)
    ax.set_ylim(0.70, 1.02)
    ax.set_xlim(-0.55, len(models) - 0.45)
    clean(ax)
    # 图例放到坐标区下方，避开柱顶的数值与 ▽ 标记
    ax.legend(frameon=False, fontsize=10.5, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.10))
    ax.set_title("三种特征方案 AUC 对照（▽ = 论文表 4 数值）",
                 fontsize=14, fontweight="bold", color=NAVY, pad=12)
    save(fig, "auc_comparison.png")


def fig_fusion_identical():
    """融合 = 深度，逐位相同。"""
    d = read_metrics()
    models = ["随机森林", "支持向量机", "XGBoost", "多层感知器"]

    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    y = np.arange(len(models))
    deep = [d[(d["建模方法"] == m) & (d["特征方案"] == "深度学习特征")]["auc"].iloc[0] for m in models]
    fus = [d[(d["建模方法"] == m) & (d["特征方案"] == "融合特征")]["auc"].iloc[0] for m in models]

    ax.barh(y + 0.19, deep, 0.34, color=BLUE, label="深度 128 维", zorder=3)
    ax.barh(y - 0.19, fus, 0.34, color=NAVY, label="融合 138 维", zorder=3)

    for yi, a, b in zip(y, deep, fus):
        ax.text(a + 0.008, yi + 0.19, f"{a:.3f}", va="center", fontsize=10,
                color=DGRAY)
        ax.text(b + 0.008, yi - 0.19, f"{b:.3f}", va="center", fontsize=10,
                color=DGRAY)

    # Δ 列放在分隔线右侧，避免与条末端的数值标签叠字
    ax.axvline(1.0, color="#D9D9D9", lw=1.2, zorder=1)
    ax.text(1.06, 3.62, "融合 − 深度", va="center", ha="left", fontsize=10,
            color=GRAY)
    for yi, a, b in zip(y, deep, fus):
        ax.text(1.06, yi, f"{b - a:+.3f}", va="center", ha="left",
                fontsize=11.5, fontweight="bold", color=RED)

    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=12)
    ax.set_xlim(0, 1.42)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_ylim(-0.7, 3.75)
    ax.set_xlabel("AUC", fontsize=12)
    clean(ax, grid_axis="x")
    ax.legend(frameon=False, fontsize=11, loc="upper center",
              bbox_to_anchor=(0.36, -0.20), ncol=2)
    ax.set_title("融合 vs 深度：4 个模型上差异全部为 0", fontsize=14,
                 fontweight="bold", color=NAVY, pad=12)
    save(fig, "fusion_identical.png")


# ======================================================================
# P13 六组检验
# ======================================================================
def fig_fusion_tests():
    """六组检验的关键数字，2×3 卡片。"""
    fig, axes = plt.subplots(2, 3, figsize=(11.6, 5.4))
    axes = axes.ravel()

    CARDS = [
        ("A · 信息冗余度", "+0.209", "平均 R²", "打乱基线 −0.075",
         "部分冗余存在，\n但不足以解释零增益", BLUE),
        ("B · 逐折配对", "+0.001~+0.007", "平均提升", "逐折标准差 ±0.08~0.10",
         "提升比噪声\n小一个数量级", BLUE),
        ("C · 错误重叠", "0.17~0.28", "重叠率", "净收益 +3，p = 0.689",
         "互补性确实存在，\n但未转化为增益", ORANGE),
        ("D · 加权拼接", "w = 0", "MLP 最优权重", "此时几何块被乘成全零",
         "暴露噪声底：\n那 +0.012 是噪声", RED),
        ("E · 概率集成", "+0.007", "最大提升", "最优 w 为事后挑选",
         "有选择性偏差，\n真实增益只会更小", ORANGE),
        ("F · 换 61 维几何", "−0.019", "随机森林变化", "189 维仍然无增益",
         "结论对几何特征\n数量不敏感", RED),
    ]

    for ax, (title, big, sub, detail, concl, color) in zip(axes, CARDS):
        ax.axis("off")
        ax.add_patch(FancyBboxPatch((0.02, 0.03), 0.96, 0.94,
                                    boxstyle="round,pad=0.02",
                                    fc="#F7F9FC", ec=color, lw=1.8,
                                    transform=ax.transAxes))
        ax.text(0.5, 0.93, title, ha="center", va="top", fontsize=12,
                fontweight="bold", color=color, transform=ax.transAxes)
        ax.text(0.5, 0.68, big, ha="center", va="center", fontsize=21,
                fontweight="bold", color=color, transform=ax.transAxes)
        ax.text(0.5, 0.51, sub, ha="center", va="center", fontsize=10,
                color=GRAY, transform=ax.transAxes)
        ax.text(0.5, 0.38, detail, ha="center", va="top", fontsize=9.5,
                color=DGRAY, transform=ax.transAxes)
        ax.text(0.5, 0.20, concl, ha="center", va="top", fontsize=10,
                color=DGRAY, style="italic", transform=ax.transAxes)

    fig.suptitle("六组独立检验：融合增益均落在噪声范围内",
                 fontsize=15, fontweight="bold", color=NAVY, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "fusion_tests.png")


def fig_weighted_scan():
    """检验 D：w 扫描曲线 —— 最优落在 w=0，暴露噪声底。

    CSV 是宽表：每行一个模型，列为「纯深度」+ 各 w 取值。
    w = 0 表示几何块被乘成全零，此时模型只看到深度特征 + 10 个常数零列。
    """
    d = pd.read_csv(RESULTS / "weighted_concat.csv", encoding="utf-8-sig")
    ws = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    wcols = ["w=0", "w=0.1", "w=0.2", "w=0.3", "w=0.4",
             "w=0.5", "w=0.6", "w=0.8", "w=1"]

    model_colors = {"随机森林": GRAY, "支持向量机": BLUE, "多层感知器": RED}

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for _, row in d.iterrows():
        name = row["建模方法"]
        col = model_colors.get(name, BLUE)
        base = float(row["纯深度"])
        ys = [float(row[c]) for c in wcols]

        # 纯深度基准线（水平虚线）
        ax.axhline(base, color=col, ls=":", lw=1.7, alpha=0.55, zorder=2)
        ax.plot(ws, ys, "o-", ms=5.5, lw=2.4, color=col, zorder=4,
                label=f"{name}（纯深度基准 {base:.3f}）")

    # 高亮 w=0 处：MLP 的「最优」恰好在这里
    mlp = d[d["建模方法"] == "多层感知器"].iloc[0]
    ax.plot([0], [float(mlp["w=0"])], "o", ms=14, mfc="none",
            mec=RED, mew=2.6, zorder=6)

    # 注释放在 RF(0.906) 与 MLP(0.940) 两条线之间的空白带里，避免压线
    ax.annotate("w = 0：几何块被乘成全零，MLP 反而在此「最优」\n→ 这 +0.012 是噪声，不是增益",
                xy=(0.03, float(mlp["w=0"]) - 0.004),
                xytext=(0.30, 0.9285),
                arrowprops=dict(arrowstyle="->", color=RED, lw=2,
                                connectionstyle="arc3,rad=-0.25"),
                fontsize=10.5, color=RED, fontweight="bold", va="top",
                bbox=dict(boxstyle="round,pad=0.35", fc="white",
                          ec=RED, lw=1.4, alpha=0.97))

    ax.set_xlabel("几何分支权重 w", fontsize=12)
    ax.set_ylabel("AUC", fontsize=12)
    ax.set_xlim(-0.06, 1.08)
    ax.set_ylim(0.880, 1.000)
    clean(ax)
    # 图例移出坐标区，避免压住曲线与注释
    ax.legend(frameon=False, fontsize=10, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.13))
    ax.set_title("检验 D：加权拼接的 w 扫描（暴露噪声底）", fontsize=14,
                 fontweight="bold", color=NAVY, pad=12)
    save(fig, "weighted_scan.png")


# ======================================================================
# P15 模型效果
# ======================================================================
def compute_oof():
    """重算 SVM + 128 维的 out-of-fold 概率（用于混淆矩阵与阈值曲线）。"""
    sys.path.insert(0, str(ROOT / "src"))
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    emb = pd.read_csv(DATA / "embeddings_128.csv", encoding="utf-8-sig")
    cols = [c for c in emb.columns if c.startswith("e") and c[1:].isdigit()]
    X = emb[cols].to_numpy(float)
    LABEL_MAP = {"digeorge": 0, "down": 1, "williams": 2}
    y = emb["label"].map(LABEL_MAP).to_numpy()
    g = emb["subject_id"].to_numpy()

    oof = np.zeros((len(y), 3))
    cv = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    for tr, te in cv.split(X, y, g):
        sc = StandardScaler().fit(X[tr])
        m = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                random_state=42).fit(sc.transform(X[tr]), y[tr])
        oof[te] = m.predict_proba(sc.transform(X[te]))
    return y, oof


def fig_confusion():
    """重绘混淆矩阵，配色与 PPT 统一。"""
    from sklearn.metrics import confusion_matrix

    y, oof = compute_oof()
    cm = confusion_matrix(y, oof.argmax(axis=1))
    names = ["DiGeorge", "唐氏", "Williams"]

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    norm = cm / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)

    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]*100:.1f}%",
                    ha="center", va="center", fontsize=13,
                    fontweight="bold" if i == j else "normal",
                    color="white" if norm[i, j] > 0.55 else DGRAY)

    ax.set_xticks(range(3))
    ax.set_xticklabels(names, fontsize=12)
    ax.set_yticks(range(3))
    ax.set_yticklabels(names, fontsize=12)
    ax.set_xlabel("预测", fontsize=12, labelpad=10)
    ax.set_ylabel("真实", fontsize=12)

    # 标注主要错误方向：写到矩阵下方的留白区，避免压住标题
    ax.add_patch(plt.Rectangle((-0.5, -0.5), 3, 3, fill=False,
                               ec="#BFBFBF", lw=1))
    ax.set_ylim(2.95, -0.5)
    ax.annotate("错误集中在「→ 唐氏」方向（8 + 6 = 14 例）",
                xy=(1.0, 2.5), xytext=(2.45, 2.72),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.8),
                fontsize=10, color=RED, fontweight="bold",
                ha="right", va="center")

    ax.set_title("混淆矩阵（10 折 out-of-fold 合并）", fontsize=14,
                 fontweight="bold", color=NAVY, pad=12)
    save(fig, "confusion.png")


def fig_threshold():
    """拒识阈值权衡曲线。"""
    from sklearn.metrics import accuracy_score

    y, oof = compute_oof()
    top = oof.max(axis=1)
    pred = oof.argmax(axis=1)

    ts = np.linspace(0.30, 0.95, 40)
    reject, acc = [], []
    for t in ts:
        keep = top >= t
        reject.append((~keep).mean())
        acc.append(accuracy_score(y[keep], pred[keep]) if keep.sum() else np.nan)

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.plot(ts, acc, "-", lw=2.6, color=NAVY, label="拒识后准确率", zorder=4)
    ax.plot(ts, reject, "--", lw=2.2, color=ORANGE, label="拒识比例", zorder=4)

    # 默认阈值
    keep = top >= 0.6
    a60 = accuracy_score(y[keep], pred[keep])
    ax.axvline(0.6, color=RED, ls=":", lw=2, zorder=5)
    ax.plot([0.6], [a60], "o", ms=11, mfc="none", mec=RED, mew=2.2, zorder=6)
    ax.annotate(f"默认阈值 0.60\n拒识 19.7%，留下 0.872",
                xy=(0.6, a60), xytext=(0.33, 0.97),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.6),
                fontsize=10.5, color=RED, fontweight="bold")

    ax.set_xlabel("拒识阈值", fontsize=12)
    ax.set_ylabel("比例", fontsize=12)
    ax.set_ylim(0, 1.05)
    clean(ax)
    ax.legend(frameon=False, fontsize=10.5, loc="lower left")
    ax.set_title("拒识阈值的权衡：越保守，留下的越可信", fontsize=14,
                 fontweight="bold", color=NAVY, pad=12)
    save(fig, "threshold_tradeoff.png")


# ======================================================================
# P18 甘特图
# ======================================================================
def fig_gantt():
    """下一步工作计划甘特图。"""
    tasks = [
        ("算法链路（已完成）", 1, 8, GREEN),
        ("SHAP 可解释性分析", 9, 2, BLUE),
        ("面部关注区域可视化", 10, 2, BLUE),
        ("FastAPI 后端 + 数据库", 9, 4, NAVY),
        ("Vue 前端界面", 11, 4, NAVY),
        ("联调 / 测试 / 文档", 14, 2, ORANGE),
    ]

    fig, ax = plt.subplots(figsize=(9.6, 3.9))
    for i, (name, start, dur, color) in enumerate(tasks):
        y = len(tasks) - 1 - i
        ax.barh(y, dur, left=start, height=0.55, color=color, zorder=3)
        ax.text(start + dur / 2, y, f"{dur} 周", ha="center", va="center",
                color="white", fontsize=10, fontweight="bold", zorder=4)

    # 中期分界线
    ax.axvline(8.5, color=RED, ls="--", lw=2, zorder=5)
    ax.text(8.65, len(tasks) - 0.35, "中期", color=RED, fontsize=12,
            fontweight="bold", va="top")

    # 里程碑（菱形与文字之间留出间距，避免压字）
    for wk, label in [(11, "M1"), (13, "M2"), (15, "M3"), (16, "M4")]:
        ax.plot([wk], [-0.82], "D", ms=9, color=NAVY, zorder=5)
        ax.text(wk, -1.32, label, ha="center", va="center", fontsize=10.5,
                fontweight="bold", color=NAVY)

    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([t[0] for t in tasks][::-1], fontsize=11.5)
    ax.set_xlim(0, 17)
    ax.set_ylim(-1.62, len(tasks) - 0.3)
    ax.set_xticks(range(0, 18, 2))
    ax.set_xlabel("周次", fontsize=12)
    clean(ax, grid_axis="x")
    ax.set_title("下一步工作计划", fontsize=14, fontweight="bold",
                 color=NAVY, pad=12)
    save(fig, "gantt.png")


# ======================================================================
def main():
    print(f"输出目录: {OUT.relative_to(ROOT)}\n")

    print("P7 数据集")
    fig_dataset_bar()
    fig_detect_funnel()
    fig_detect_rate()

    print("P8 几何特征")
    fig_landmarks()
    fig_geo_dist()

    print("P9 深度特征")
    fig_desc_distance()

    print("P10 筛选与融合")
    fig_importance()
    fig_fusion_138()

    print("P12 实验结果")
    fig_auc_comparison()
    fig_fusion_identical()

    print("P13 六组检验")
    fig_fusion_tests()
    fig_weighted_scan()

    print("P15 模型效果")
    fig_confusion()
    fig_threshold()

    print("P18 计划")
    fig_gantt()

    print(f"\n完成，共 {len(list(OUT.glob('*.png')))} 张图")
    for p in sorted(OUT.glob("*.png")):
        print(f"  {p.name:<26} {p.stat().st_size / 1024:>6.0f} KB")


if __name__ == "__main__":
    main()
