# -*- coding: utf-8 -*-
"""
按 docs/MIDTERM_PPT.md 生成中期报告 PPT（19 页，16:9，含切换与进入动画）。

    PYTHONIOENCODING=utf-8 python src/make_ppt.py

依赖 docs/ppt_assets/ 下的插图（先跑 src/make_ppt_figures.py）。
"""

from pathlib import Path
import sys

from pptx import Presentation
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches, Pt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ppt_kit import (  # noqa: E402
    BG, BLUE, DGRAY, FONT, FONT_NUM, GRAY, GREEN, HEADER_H, LBLUE, LGRAY,
    MARGIN, NAVY, ORANGE, PALE, RED, SLIDE_H, SLIDE_W, WHITE,
    add_cell, add_header, add_image_fit, add_rect, add_table, add_text,
    animate, content_slide, fade_all,
)

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "docs" / "ppt_assets"
OUT_FILE = ROOT / "docs" / "midterm_report.pptx"

TITLE = "基于面部图像的罕见病诊断算法与系统"
MEMBERS = "梁顺健 ｜ 张锦霖 ｜ 王家柯 ｜ 蒋云浩"

CENTER = PP_ALIGN.CENTER
RIGHT = PP_ALIGN.RIGHT


# ======================================================================
# P1 封面
# ======================================================================
def p01_cover(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_rect(s, 0, 0, SLIDE_W, SLIDE_H, fill=NAVY)
    # 装饰色带
    add_rect(s, 0, 0, SLIDE_W, 0.10, fill=BLUE)
    add_rect(s, 0, SLIDE_H - 0.10, SLIDE_W, 0.10, fill=BLUE)

    a1 = add_text(s, 1.0, 1.15, 11.3, 0.4,
                  "哈尔滨工业大学（深圳） · 面向领域的计算机系统设计与开发实践",
                  size=15, color=LBLUE, align=CENTER)
    a2 = add_text(s, 1.0, 1.62, 11.3, 0.4, "中 期 报 告", size=16,
                  color=LBLUE, align=CENTER, bold=True)

    a3 = add_text(s, 0.8, 2.55, 11.7, 1.3, TITLE,
                  size=40, color=WHITE, bold=True, align=CENTER, spacing=1.1)

    add_rect(s, 5.42, 4.05, 2.5, 0.045, fill=BLUE)

    a4 = add_text(s, 1.0, 4.42, 11.3, 0.42,
                  "组号 3 ｜ 班号 18 ｜ 计算机科学与技术学院",
                  size=15, color=LBLUE, align=CENTER)
    a5 = add_text(s, 1.0, 4.92, 11.3, 0.42, MEMBERS,
                  size=15, color=WHITE, align=CENTER)

    a6 = add_text(s, 1.0, 6.02, 11.3, 0.4,
                  "汇报人：＿＿＿＿＿＿          2026 年 ＿＿ 月 ＿＿ 日",
                  size=13.5, color=LBLUE, align=CENTER)

    animate(s, [
        fade_all(a1, effect="fade", dur=400),
        fade_all(a2, effect="fade", dur=400),
        fade_all(a3, effect="dissolve", dur=700),
        fade_all(a4, a5, effect="wipe_up", dur=500),
        fade_all(a6, effect="fade", dur=400),
    ])
    return s


# ======================================================================
# P2 汇报提纲
# ======================================================================
def p02_outline(prs):
    s = content_slide(prs, "汇报提纲", 2, kicker="CONTENTS")
    items = [
        ("01", "课题背景与技术路线", "问题背景、应用意义\n任务书要求、总体方案"),
        ("02", "已完成工作", "数据、特征、模型、实验\n可调用模型封装"),
        ("03", "关键实验发现", "融合特征增益的证伪\n与六组独立检验"),
        ("04", "问题与计划", "当前困难、方案调整\n下一步安排与分工"),
    ]
    cards = []
    x = MARGIN
    w = 2.85
    for num, title, desc in items:
        box = add_rect(s, x, 2.05, w, 3.35, fill=BG, line=LGRAY,
                       shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
        add_text(s, x + 0.30, 2.42, w - 0.6, 0.9, num, size=46, color=LBLUE,
                 bold=True, font=FONT_NUM)
        add_text(s, x + 0.30, 3.42, w - 0.6, 0.75, title, size=17, color=NAVY,
                 bold=True)
        add_text(s, x + 0.30, 4.22, w - 0.6, 1.0, desc, size=11.5, color=GRAY,
                 spacing=1.35)
        cards.append(box)
        x += w + 0.30

    tip = add_text(s, MARGIN, 5.80, 12.29, 0.5,
                   "第 03 部分是本次中期最值得关注的结论：对一个已发表算法主张的"
                   "可复现性检验。",
                   size=13, color=RED, align=CENTER, bold=True)

    animate(s, [fade_all(*cards, effect="fade", dur=500),
                fade_all(tip, effect="fade", dur=400)])
    return s


# ======================================================================
# P3 课题背景与意义
# ======================================================================
def p03_background(prs):
    s = content_slide(prs, "课题背景与应用意义", 3, kicker="01 · 背景")

    # ---- 左栏 ----
    lt = add_text(s, MARGIN, 1.42, 6.0, 0.4, "问题背景", size=18,
                  color=NAVY, bold=True)
    lb = add_text(s, MARGIN, 1.98, 6.0, 3.6, [
        [("①  ", {"color": BLUE, "bold": True}),
         ("罕见病患病人数少、种类多、临床表现复杂", {}),
         ("，约 7000 种，单病种发病率极低。", {})],
        [("②  ", {"color": BLUE, "bold": True}),
         ("相当一部分遗传性综合征具有特征性面部表型", {}),
         ("——眼距、鼻梁、唇形等，可作为识别线索。", {})],
        [("③  ", {"color": BLUE, "bold": True}),
         ("传统诊断依赖多项检查与医生经验", {}),
         ("，流程复杂、周期长，基层难以覆盖。", {})],
    ], size=14, spacing=1.5, space_after=14)

    # ---- 右栏 ----
    rt = add_text(s, 6.85, 1.42, 5.95, 0.4, "应用意义", size=18,
                  color=NAVY, bold=True)
    cards = []
    for i, (t, d) in enumerate([
        ("降低主观性", "机器量化面部几何结构，\n减少人工观察的个体差异"),
        ("提高初筛效率", "在基因检测前给出概率性提示，\n辅助排序可疑病种"),
        ("结果可解释", "展示判断依据与关注区域，\n而非只给黑箱结论"),
    ]):
        y = 1.98 + i * 1.18
        b = add_rect(s, 6.85, y, 5.95, 1.02, fill=BG, line=LBLUE,
                     shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.10)
        add_text(s, 7.10, y + 0.13, 2.1, 0.35, t, size=14, color=BLUE,
                 bold=True)
        add_text(s, 7.10, y + 0.50, 5.45, 0.48, d.replace("\n", "  "),
                 size=11.5, color=DGRAY)
        cards.append(b)

    band = add_rect(s, MARGIN, 5.72, 12.29, 0.95, fill=BG, line=NAVY,
                    line_w=1.5, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.10)
    add_text(s, MARGIN + 0.35, 5.90, 11.6, 0.62, [
        [("本课题定位：", {"bold": True, "color": NAVY}),
         ("面向具有典型面部表型的罕见遗传综合征，做「算法研究 + 系统实现」"
          "的完整工程实践，", {}),
         ("而非临床诊断产品。", {"bold": True, "color": RED})],
    ], size=13.5, spacing=1.25)

    animate(s, [
        fade_all(lt, lb, effect="wipe_right", dur=500),
        [(cards[0], "wipe_right", 0, 450)],
        [(cards[1], "wipe_right", 0, 450)],
        [(cards[2], "wipe_right", 0, 450)],
        fade_all(rt, effect="fade", dur=350),
        fade_all(band, effect="fade", dur=500),
    ])
    return s


# ======================================================================
# P4 任务书要求回顾
# ======================================================================
def p04_taskbook(prs):
    s = content_slide(prs, "任务书要求回顾", 4, kicker="01 · 依据")

    t1 = add_text(s, MARGIN, 1.34, 12.29, 0.4,
                  [("任务题目：", {"color": GRAY, "size": 13}),
                   (TITLE, {"color": NAVY, "size": 17, "bold": True})])

    t2 = add_rect(s, MARGIN, 1.86, 12.29, 0.62, fill=BG, line=LBLUE,
                  shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.14)
    add_text(s, MARGIN + 0.30, 2.00, 11.7, 0.36,
             [("拟定技术路线：", {"bold": True, "color": NAVY}),
              ("面部图像预处理 → 特征提取 → 特征筛选与融合 → 分类模型训练 → "
               "模型解释 → Web 系统实现", {"color": DGRAY})],
             size=12.5)

    # 10 项核心功能：两栏各 5 行
    funcs = [
        ("1", "面部图像上传及预览", "6", "输出各类别概率及 Top-K"),
        ("2", "人脸检测、裁剪、对齐及预处理", "7", "SHAP 解释关键面部特征"),
        ("3", "关键点检测与几何特征提取", "8", "可视化模型关注的面部区域"),
        ("4", "深度特征提取及特征融合", "9", "识别历史记录查询与展示"),
        ("5", "基于 XGBoost 的辅助分类", "10", "异常情况提示"),
    ]
    add_text(s, MARGIN, 2.70, 6.0, 0.35, "Web 系统 10 项核心功能",
             size=15, color=NAVY, bold=True)

    rows = []
    for a, b, c, d in funcs:
        y = 3.14 + len(rows) * 0.55
        r1 = add_cell(s, MARGIN, y, 0.42, 0.48, a, size=12, color=WHITE,
                      bold=True, fill=BLUE, align=CENTER)
        r2 = add_cell(s, MARGIN + 0.42, y, 5.55, 0.48, b, size=12,
                      fill=WHITE)
        r3 = add_cell(s, 6.85, y, 0.42, 0.48, c, size=12, color=WHITE,
                      bold=True, fill=BLUE, align=CENTER)
        r4 = add_cell(s, 6.85 + 0.42, y, 5.55, 0.48, d, size=12, fill=WHITE)
        rows.append([r1, r2, r3, r4])

    stack = add_rect(s, MARGIN, 6.02, 12.29, 0.72, fill=BG, line=LGRAY,
                     shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.12)
    add_text(s, MARGIN + 0.30, 6.14, 11.7, 0.5,
             [("拟定技术栈：", {"bold": True, "color": NAVY}),
              ("后端 Python + FastAPI（+ Java SpringBoot）；前端 Vue；"
               "存储 SQLite / MySQL；算法侧 Dlib + XGBoost + 10 折交叉验证 + SHAP",
               {"color": DGRAY})],
             size=12)

    animate(s, [
        fade_all(t1, t2, effect="wipe_right", dur=500),
        [(r, "fade", i * 60, 300) for i, r in enumerate(rows)],
        fade_all(stack, effect="fade", dur=400),
    ])
    return s


# ======================================================================
# P5 总体技术方案
# ======================================================================
def p05_pipeline(prs):
    s = content_slide(prs, "总体技术方案", 5, kicker="01 · 路线")

    stages = [
        ("图像预处理", "检测 / 对齐 / 关键点", BLUE),
        ("特征提取", "61 维几何 + 128 维深度", BLUE),
        ("筛选与融合", "Top-10 筛选 → 138 维", GRAY),
        ("模型训练", "4 模型 × 10 折分组交叉验证", BLUE),
        ("模型解释", "SHAP + 区域可视化", BLUE),
        ("Web 系统", "FastAPI + Vue + 数据库", BLUE),
    ]
    boxes = []
    bw, gap = 1.86, 0.30
    x = MARGIN
    for i, (name, desc, col) in enumerate(stages):
        b = add_rect(s, x, 2.05, bw, 1.05, fill=col, line=None,
                     shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.10)
        add_text(s, x + 0.08, 2.26, bw - 0.16, 0.35, name, size=13,
                 color=WHITE, bold=True, align=CENTER)
        add_text(s, x + 0.06, 2.65, bw - 0.12, 0.42, desc, size=9,
                 color=WHITE, align=CENTER, spacing=1.05)
        boxes.append(b)
        if i < len(stages) - 1:
            ar = add_rect(s, x + bw + 0.03, 2.44, 0.24, 0.26, fill=LGRAY,
                          shape=MSO_SHAPE.RIGHT_ARROW)
            boxes.append(ar)
        x += bw + gap

    # 融合环节的结论标注
    note = add_rect(s, MARGIN + 2 * (bw + gap), 3.28, bw + 0.9, 0.62,
                    fill=WHITE, line=RED, line_w=1.5,
                    shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.14)
    add_text(s, MARGIN + 2 * (bw + gap) + 0.10, 3.40, bw + 0.7, 0.40,
             "⚠ 已验证：该环节增益不显著\n（见 P12–P13）",
             size=9.5, color=RED, bold=True, align=CENTER, spacing=1.1)

    # 两个贯穿全程的约束
    c1 = add_rect(s, MARGIN, 4.35, 6.0, 2.28, fill=BG, line=LBLUE,
                  shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
    add_text(s, MARGIN + 0.32, 4.58, 5.4, 1.85, [
        [("约束一 · 小样本", {"size": 15, "bold": True, "color": NAVY})],
        [("训练样本仅 117 个，类别分布 53 / 33 / 31。", {"size": 12})],
        [("→ 必须靠交叉验证与正则化控制过拟合；", {"size": 12, "color": GRAY})],
        [("→ 不追求提高模型复杂度。", {"size": 12, "color": GRAY})],
    ], spacing=1.3, space_after=7)

    c2 = add_rect(s, 6.81, 4.35, 6.0, 2.28, fill=BG, line=LBLUE,
                  shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
    add_text(s, 7.13, 4.58, 5.4, 1.85, [
        [("约束二 · 防数据泄漏", {"size": 15, "bold": True, "color": NAVY})],
        [("标准化器与特征选择器只在训练折上拟合。", {"size": 12})],
        [("→ 若在全量数据上 fit scaler，", {"size": 12, "color": GRAY})],
        [("   测试折信息会渗入训练，导致高估性能。", {"size": 12, "color": GRAY})],
    ], spacing=1.3, space_after=7)

    animate(s, [
        [(tuple(boxes), "wipe_up", 0, 550)],               # 流程整体推入
        [(note, "fade", 0, 500)],
        fade_all(c1, effect="wipe_right", dur=500),
        fade_all(c2, effect="wipe_right", dur=500),
    ])
    return s


# ======================================================================
# P6 进度总览
# ======================================================================
def p06_progress(prs):
    s = content_slide(prs, "进度总览：已完成 vs 待完成", 6, kicker="01 · 进度")

    ok, part, todo = GREEN, ORANGE, RED

    left = [
        ("数据集整理与预处理", "已完成", ok),
        ("面部检测与关键点定位", "已完成", ok),
        ("61 维几何特征提取", "已完成", ok),
        ("128 维深度特征提取", "已完成", ok),
        ("重要特征筛选（Top-10）", "已完成", ok),
        ("特征融合（138 维）", "已完成", ok),
        ("分类模型设计与对比", "已完成", ok),
        ("模型性能评价", "已完成", ok),
        ("可调用模型封装", "已完成", ok),
    ]
    right = [
        ("无人脸 / 低置信度提示", "已完成", ok),
        ("多人脸 / 低质量图像提示", "部分完成", part),
        ("输出 Top-K 预测结果", "部分完成", part),
        ("SHAP 可解释性分析", "未开始", todo),
        ("模型关注区域可视化", "未开始", todo),
        ("识别历史记录管理", "未开始", todo),
        ("Web 系统（FastAPI+Vue）", "未开始", todo),
        ("Java SpringBoot 后端", "未开始", todo),
        ("联调 / 测试 / 文档", "未开始", todo),
    ]

    def draw_table(items, x, head):
        add_text(s, x, 1.32, 6.0, 0.35, head, size=15, color=NAVY, bold=True)
        shapes = []
        hdr = [add_cell(s, x, 1.76, 4.10, 0.42, "任务项", size=12,
                        color=WHITE, bold=True, fill=NAVY),
               add_cell(s, x + 4.10, 1.76, 1.42, 0.42, "状态", size=12,
                        color=WHITE, bold=True, fill=NAVY, align=CENTER)]
        shapes.append(hdr)
        for i, (name, status, col) in enumerate(items):
            y = 2.18 + i * 0.47
            fill = PALE if i % 2 == 0 else WHITE
            r = [add_cell(s, x, y, 4.10, 0.47, name, size=11.5, fill=fill),
                 add_cell(s, x + 4.10, y, 1.42, 0.47, status, size=11.5,
                          color=col, bold=True, fill=fill, align=CENTER)]
            shapes.append(r)
        return shapes

    L = draw_table(left, MARGIN, "算法链路")
    R = draw_table(right, 6.85, "系统集成")

    band = add_rect(s, MARGIN, 6.48, 12.29, 0.50, fill=NAVY)
    add_text(s, MARGIN + 0.3, 6.58, 11.7, 0.32,
             "整体进度：算法链路 100%　｜　系统集成 0%　"
             "（算法先行、系统后置：选型结论未定前做集成会返工）",
             size=13, color=WHITE, bold=True, align=CENTER)

    animate(s, [
        fade_all(*L[0], *R[0], effect="fade", dur=350),
        [tuple(r) for r in L[1:]],                    # 左表逐行推入
        [tuple(r) for r in R[1:]],                    # 右表逐行推入
        fade_all(band, effect="wipe_up", dur=450),
    ])
    return s


# ======================================================================
# P7 数据集与预处理
# ======================================================================
def p07_dataset(prs):
    s = content_slide(prs, "数据集与预处理", 7, kicker="02 · 数据")

    i1 = add_image_fit(s, ASSETS / "dataset_bar.png", MARGIN, 1.32, 6.0, 2.72)
    i2 = add_image_fit(s, ASSETS / "detect_funnel.png", 6.81, 1.32, 6.0, 2.72)

    add_text(s, MARGIN, 4.06, 6.0, 0.32, "预处理链路", size=14,
             color=NAVY, bold=True)
    chain = add_rect(s, MARGIN, 4.44, 6.0, 1.08, fill=BG, line=LBLUE,
                     shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.10)
    add_text(s, MARGIN + 0.24, 4.56, 5.5, 0.85,
             "读图（PIL → RGB，兼容带 alpha 的 PNG）\n"
             "→ HOG 人脸检测（上采样 1 次，多脸取最大）\n"
             "→ 68 点关键点 → 5 点仿射对齐 → 128 维描述子",
             size=11, color=DGRAY, spacing=1.35)

    warn = add_rect(s, 6.81, 4.44, 6.0, 1.08, fill=WHITE, line=RED, line_w=1.6,
                    shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.10)
    add_text(s, 7.05, 4.56, 5.5, 0.85,
             "⚠ 关键约束：只适用于接近正面的人脸\n"
             "330 张中仅 127 张可检出（38%）。其余多为侧脸、\n"
             "仰角、眼部特写——这是适用范围所限，不是程序缺陷。",
             size=11, color=RED, bold=True, spacing=1.35)

    band = add_rect(s, MARGIN, 5.72, 12.29, 0.98, fill=BG, line=LGRAY,
                    shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
    add_text(s, MARGIN + 0.28, 5.84, 11.8, 0.78,
             [("已目视确认：", {"bold": True, "color": NAVY}),
              ("一张 02.jpg 是孩子的 90° 侧面照，检不出人脸是"
               "正确行为。训练集使用的 117 张全部来自可检出的正面照，"
               "数据集未选用的 10 张不计入。", {"color": DGRAY})],
             size=12, spacing=1.3)

    animate(s, [
        fade_all(i1, effect="wipe_right", dur=550),
        fade_all(i2, effect="wipe_right", dur=550),
        fade_all(chain, effect="fade", dur=450),
        fade_all(warn, effect="fade", dur=450),
        fade_all(band, effect="fade", dur=400),
    ])
    return s


# ======================================================================
# P8 几何特征
# ======================================================================
def p08_geo(prs):
    s = content_slide(prs, "特征提取（一）：61 维几何特征", 8, kicker="02 · 特征")

    img = add_image_fit(s, ASSETS / "landmarks_face.png", MARGIN, 1.30, 7.55, 3.60)
    comp = add_image_fit(s, ASSETS / "geo_composition.png", 8.25, 1.30, 4.56, 3.60)

    band = add_rect(s, MARGIN, 5.16, 12.29, 1.56, fill=BG, line=LBLUE,
                    shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
    add_text(s, MARGIN + 0.32, 5.32, 11.7, 1.28, [
        [("为什么用这 61 维", {"size": 13.5, "bold": True, "color": NAVY})],
        [("35 个距离 + 26 个角度，全部按「两外眼角间距 d(0,16)」或人脸宽 L "
          "归一化，角度取 |cos θ|。", {"size": 11.5})],
        [("→ 尺度不变：与图像分辨率、拍摄距离无关，同一张脸在任何尺寸下算出的"
          "几何特征一致。", {"size": 11.5, "color": GREEN})],
        [("→ 可解释：每一个维度都能指回脸部的具体结构（鼻尖到右眼、鼻梁长度、"
          "下唇长……）。", {"size": 11.5, "color": GREEN})],
    ], spacing=1.30, space_after=4)

    animate(s, [
        fade_all(img, effect="dissolve", dur=700),
        fade_all(comp, effect="wipe_right", dur=600),
        fade_all(band, effect="wipe_up", dur=500),
    ])
    return s


# ======================================================================
# P9 深度特征
# ======================================================================
def p09_deep(prs):
    s = content_slide(prs, "特征提取（二）：128 维深度特征", 9, kicker="02 · 特征")

    add_text(s, MARGIN, 1.32, 5.9, 0.35, "Dlib 描述子的三级流水线", size=15,
             color=NAVY, bold=True)

    steps = [
        ("① 人脸检测", "get_frontal_face_detector", "RGB 图像 → 人脸方框"),
        ("② 关键点定位", "shape_predictor_68", "图像 + 方框 → 68 点"),
        ("③ 描述子提取", "resnet_model_v1", "图像 + 68 点 → 128 维"),
    ]
    boxes = []
    for i, (name, model, io) in enumerate(steps):
        y = 1.78 + i * 1.02
        b = add_rect(s, MARGIN, y, 5.9, 0.88, fill=BG, line=BLUE,
                     shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.10)
        add_text(s, MARGIN + 0.24, y + 0.10, 2.0, 0.32, name, size=13,
                 color=NAVY, bold=True)
        add_text(s, MARGIN + 2.30, y + 0.10, 3.4, 0.32, model, size=10,
                 color=BLUE, font="Consolas")
        add_text(s, MARGIN + 0.24, y + 0.48, 5.4, 0.30, io, size=11,
                 color=GRAY)
        boxes.append(b)

    why = add_rect(s, MARGIN, 4.95, 5.9, 1.68, fill=WHITE, line=LGRAY,
                   shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
    add_text(s, MARGIN + 0.26, 5.08, 5.4, 1.45, [
        [("为什么关键点必须传给第 ③ 步", {"size": 12.5, "bold": True, "color": RED})],
        [("网络内部会挑 5 个点（两眼中点、鼻尖、两嘴角）", {"size": 11})],
        [("做仿射对齐，先把人脸摆正再送进 ResNet。", {"size": 11})],
        [("结构：ResNet 卷积层 + 全局平均池化 → 128 维。", {"size": 11, "color": GRAY})],
    ], spacing=1.3, space_after=6)

    dist = add_image_fit(s, ASSETS / "desc_distance.png", 6.81, 1.70, 6.0, 3.1)

    concl = add_rect(s, 6.81, 4.95, 6.0, 1.68, fill=BG, line=GREEN, line_w=1.5,
                     shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
    add_text(s, 7.07, 5.10, 5.5, 1.45, [
        [("✅ 质量校验通过", {"size": 13, "bold": True, "color": GREEN})],
        [("同人 0.350 vs 异人 0.641，区分度 0.291；", {"size": 11.5})],
        [("dlib 官方阈值 0.6 下同人命中率 100%。", {"size": 11.5})],
        [("说明关键点与原图对齐正确，描述子有效。", {"size": 11.5, "color": GRAY})],
    ], spacing=1.3, space_after=6)

    animate(s, [
        [(b, "wipe_right", i * 150, 450) for i, b in enumerate(boxes)],
        fade_all(why, effect="fade", dur=450),
        fade_all(dist, effect="dissolve", dur=650),
        fade_all(concl, effect="fade", dur=450),
    ])
    return s


# ======================================================================
# P10 特征筛选与融合
# ======================================================================
def p10_select(prs):
    s = content_slide(prs, "特征筛选与融合", 10, kicker="02 · 特征")

    imp = add_image_fit(s, ASSETS / "feature_importance.png", MARGIN, 1.30, 7.4, 4.05)

    card = add_rect(s, 8.20, 1.36, 4.61, 1.72, fill=BG, line=GREEN, line_w=1.5,
                    shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
    add_text(s, 8.44, 1.50, 4.15, 1.5, [
        [("✅ 实现正确性的有力证据", {"size": 12.5, "bold": True, "color": GREEN})],
        [("本次 Top 5 全部落在论文 Top 10 内，", {"size": 11})],
        [("其中 4 个命中论文 Top 5。", {"size": 11})],
    ], spacing=1.35, space_after=5)

    note = add_rect(s, 8.20, 3.22, 4.61, 2.13, fill=WHITE, line=LGRAY,
                    shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
    add_text(s, 8.44, 3.36, 4.15, 1.9, [
        [("论文 Top 10 中另 5 项", {"size": 11.5, "bold": True, "color": NAVY})],
        [("（左眼宽、上唇相关）本次排到 16–49 名，", {"size": 10.5, "color": GRAY})],
        [("已定性为数据依赖性差异，", {"size": 10.5, "color": GRAY})],
        [("非实现错误。", {"size": 10.5, "color": GRAY})],
    ], spacing=1.3, space_after=5)

    # 融合条：用原生形状画，缩放不失真（比例 10 : 128 悬殊，用文字标注实际维度）
    lab = add_text(s, MARGIN, 5.38, 6.0, 0.30, "融合特征构建", size=13,
                   color=NAVY, bold=True)

    def block(x, w, fill, text, color=WHITE, size=11.5, h=0.78):
        b = add_rect(s, x, 5.72, w, h, fill=fill,
                     shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.10)
        add_text(s, x + 0.10, 5.86, w - 0.20, h - 0.28, text, size=size,
                 color=color, bold=True, align=CENTER)
        return b

    b1 = block(MARGIN, 1.62, BLUE, "Top-10\n几何特征", size=10.5)
    op1 = add_text(s, 2.20, 5.88, 0.34, 0.46, "+", size=20, color=GRAY,
                   bold=True, align=CENTER)
    b2 = block(2.60, 6.34, NAVY, "Dlib 深度特征  128 维")
    op2 = add_text(s, 9.00, 5.88, 0.34, 0.46, "=", size=20, color=GRAY,
                   bold=True, align=CENTER)
    b3 = block(9.40, 3.41, GREEN, "融合特征  138 维")

    sub = add_text(s, MARGIN + 0.02, 6.56, 12.27, 0.28,
                   "10 + 128 = 138　　比例悬殊，几何分支在总维度中仅占 7.2%",
                   size=10, color=GRAY)

    animate(s, [
        fade_all(imp, effect="wipe_right", dur=600),
        fade_all(card, effect="fade", dur=450),
        fade_all(note, effect="fade", dur=450),
        [(lab, "fade", 0, 350)],
        [(b1, "wipe_right", 0, 400)],
        [(op1, "fade", 0, 250), (b2, "wipe_right", 0, 450)],
        [(op2, "fade", 0, 250), (b3, "wipe_right", 0, 400)],
        [(sub, "fade", 0, 350)],
    ])
    return s


# ======================================================================
# P11 分类模型与实验设计
# ======================================================================
def p11_design(prs):
    s = content_slide(prs, "分类模型与实验设计", 11, kicker="02 · 方法")

    models = [
        ("随机森林", "n_estimators=100"),
        ("支持向量机", "kernel=rbf, C=1.0"),
        ("XGBoost", "默认参数"),
        ("多层感知器", "默认参数"),
    ]
    cards = []
    w, gap = 2.92, 0.20
    for i, (name, param) in enumerate(models):
        x = MARGIN + i * (w + gap)
        b = add_rect(s, x, 1.32, w, 1.10, fill=NAVY if i == 1 else BG,
                     line=BLUE if i == 1 else LGRAY,
                     shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.10)
        add_text(s, x + 0.20, 1.48, w - 0.4, 0.35, name, size=14,
                 color=WHITE if i == 1 else NAVY, bold=True, align=CENTER)
        add_text(s, x + 0.16, 1.92, w - 0.32, 0.35, param, size=9.5,
                 color=LBLUE if i == 1 else GRAY, align=CENTER,
                 font="Consolas")
        cards.append(b)

    calc = add_rect(s, MARGIN, 2.62, 12.29, 0.62, fill=PALE, line=BLUE,
                    shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.14)
    add_text(s, MARGIN + 0.30, 2.75, 11.7, 0.40,
             [("实验规模：", {"bold": True, "color": NAVY}),
              ("3 种特征方案（几何 10 / 深度 128 / 融合 138）× 4 个模型 × "
               "10 折交叉验证 = ", {"color": DGRAY}),
              ("120 次训练", {"bold": True, "color": RED})],
             size=13)

    add_text(s, MARGIN, 3.42, 12.29, 0.35,
             "⚠ 两个关键设计（与任务书的差异，均属改进）", size=15,
             color=NAVY, bold=True)

    hdr = [add_cell(s, MARGIN, 3.86, 2.30, 0.44, "设计点", size=12,
                    color=WHITE, bold=True, fill=NAVY),
           add_cell(s, MARGIN + 2.30, 3.86, 2.30, 0.44, "任务书", size=12,
                    color=WHITE, bold=True, fill=NAVY, align=CENTER),
           add_cell(s, MARGIN + 4.60, 3.86, 2.69, 0.44, "本课题实现", size=12,
                    color=WHITE, bold=True, fill=NAVY, align=CENTER),
           add_cell(s, MARGIN + 7.29, 3.86, 5.00, 0.44, "理由", size=12,
                    color=WHITE, bold=True, fill=NAVY)]
    rows = []
    for i, (a, b, c, d) in enumerate([
        ("折划分", "「分层 10 折」", "StratifiedGroupKFold",
         "同一受试者不跨折；否则同一人的照片会分处训练/测试折，造成信息泄漏、高估性能"),
        ("标准化与筛选", "未说明", "只在训练折上拟合",
         "若在全量数据上 fit scaler，测试折信息会渗入训练"),
    ]):
        y = 4.30 + i * 0.78
        fill = PALE if i % 2 == 0 else WHITE
        r = [add_cell(s, MARGIN, y, 2.30, 0.78, a, size=11.5, fill=fill,
                      bold=True, color=NAVY),
             add_cell(s, MARGIN + 2.30, y, 2.30, 0.78, b, size=11, fill=fill,
                      color=GRAY, align=CENTER),
             add_cell(s, MARGIN + 4.60, y, 2.69, 0.78, c, size=11, fill=fill,
                      color=GREEN, bold=True, align=CENTER),
             add_cell(s, MARGIN + 7.29, y, 5.00, 0.78, d, size=10.5,
                      fill=fill)]
        rows.append(r)

    metric = add_rect(s, MARGIN, 5.98, 12.29, 0.78, fill=WHITE, line=LGRAY,
                      shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.10)
    add_text(s, MARGIN + 0.30, 6.10, 11.7, 0.55,
             [("评价指标：", {"bold": True, "color": NAVY}),
              ("精确率 / 召回率 / F1（macro 宏平均）、AUC（micro + OVR）、"
               "ROC 曲线、混淆矩阵", {"color": DGRAY})],
             size=12.5)

    animate(s, [
        [(b, "fade", i * 120, 400) for i, b in enumerate(cards)],
        fade_all(calc, effect="wipe_up", dur=450),
        fade_all(hdr[0], hdr[1], hdr[2], hdr[3], effect="fade", dur=350),
        [tuple(r) for r in rows],
        fade_all(metric, effect="fade", dur=400),
    ])
    return s


# ======================================================================
# P12 实验结果
# ======================================================================
def p12_results(prs):
    s = content_slide(prs, "实验结果：三种特征方案对照", 12, kicker="02 · 结果")

    auc = add_image_fit(s, ASSETS / "auc_comparison.png", MARGIN, 1.26, 7.85, 4.05)

    ident = add_image_fit(s, ASSETS / "fusion_identical.png", 8.55, 1.26,
                          4.26, 4.05)

    band = add_rect(s, MARGIN, 5.48, 12.29, 1.28, fill=NAVY)
    add_text(s, MARGIN + 0.42, 5.62, 11.5, 1.0, [
        [("核心结论：", {"size": 15, "bold": True, "color": WHITE}),
         ("论文核心主张「融合后 128 → 138 维进一步提升分类效果」",
          {"size": 14, "color": WHITE}),
         (" 在本地数据上未复现。", {"size": 14, "bold": True, "color": LBLUE})],
        [("几何列与深度列与论文基本吻合；但融合列与深度列在 4 个模型上"
          "AUC 逐位相同，连小幅波动都没有。", {"size": 12, "color": LBLUE})],
    ], spacing=1.35, space_after=8)

    animate(s, [
        fade_all(auc, effect="wipe_right", dur=650),
        fade_all(ident, effect="dissolve", dur=650),
        fade_all(band, effect="wipe_up", dur=550),
    ])
    return s


# ======================================================================
# P13 六组检验（核心页）
# ======================================================================
def p13_tests(prs):
    s = content_slide(prs, "关键发现：融合增益的证伪", 13, kicker="03 · 核心")

    head = add_rect(s, MARGIN, 1.24, 12.29, 0.60, fill=NAVY)
    add_text(s, MARGIN + 0.3, 1.37, 11.7, 0.36,
             "结论：融合特征相比纯深度特征无显著增益，差异全部落在噪声范围内",
             size=15, color=WHITE, bold=True, align=CENTER)

    # 两张图并排：各自独立占满半幅，比上下堆叠放大得多
    scan = add_image_fit(s, ASSETS / "weighted_scan.png", MARGIN, 1.92,
                         6.05, 3.30)
    tests = add_image_fit(s, ASSETS / "fusion_tests.png", 6.72, 1.92,
                          6.09, 3.30)

    why = add_rect(s, MARGIN, 5.32, 12.29, 1.40, fill=BG, line=RED, line_w=1.5,
                   shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
    add_text(s, MARGIN + 0.30, 5.44, 2.5, 0.35, "为什么没有增益？", size=13,
             color=RED, bold=True)
    cards = []
    for i, (t, d) in enumerate([
        ("分支强弱悬殊", "深度 0.93–0.95，几何 0.78–0.82。\n"
                        "弱分支能补的空间本就有限。"),
        ("互补样本太少", "两分支出错样本重合度仅 0.17–0.28，\n"
                        "但真正互补的样本全数据集只有约 9 个。"),
        ("样本量过小", "逐折标准差 ±0.08~0.10，\n"
                      "任何 ≤0.007 的提升都淹没在噪声里。"),
    ]):
        x = MARGIN + 0.30 + i * 3.92
        add_text(s, x, 5.82, 3.60, 0.30, f"{i+1}. {t}", size=11.5,
                 color=NAVY, bold=True)
        add_text(s, x + 0.16, 6.14, 3.50, 0.50, d, size=10, color=DGRAY,
                 spacing=1.25)
        cards.append(t)

    animate(s, [
        fade_all(head, effect="wipe_up", dur=450),
        fade_all(scan, effect="dissolve", dur=700),
        fade_all(tests, effect="wipe_right", dur=700),
        fade_all(why, effect="fade", dur=500),
    ])
    return s


# ======================================================================
# P14 阶段性交付
# ======================================================================
def p14_deliver(prs):
    s = content_slide(prs, "阶段性交付：可调用的预测模型", 14, kicker="02 · 交付")

    add_text(s, MARGIN, 1.30, 6.2, 0.35, "已交付文件", size=15,
             color=NAVY, bold=True)
    tree = add_rect(s, MARGIN, 1.74, 6.2, 2.30, fill=BG, line=LGRAY,
                    shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
    add_text(s, MARGIN + 0.28, 1.90, 5.7, 2.05, [
        [("models/final/svm_deep.joblib", {"color": BLUE, "bold": True}),
         ("      0.12 MB  训练好的分类器", {"color": GRAY})],
        [("models/final/threshold_table.csv", {"color": BLUE, "bold": True}),
         ("         拒识阈值对照表", {"color": GRAY})],
        [("src/face_embedding.py", {"color": BLUE, "bold": True}),
         ("                共享特征底层（训练/推理同源）", {"color": GRAY})],
        [("src/train_final.py", {"color": BLUE, "bold": True}),
         ("                     模型训练与封装", {"color": GRAY})],
        [("src/predict.py", {"color": BLUE, "bold": True}),
         ("                        推理入口（CLI + Python API）", {"color": GRAY})],
        [("docs/MODEL_USAGE.md", {"color": BLUE, "bold": True}),
         ("                  模型使用说明", {"color": GRAY})],
    ], size=11, spacing=1.42, space_after=0, font="Consolas")

    code = add_rect(s, 7.05, 1.74, 5.76, 2.30, fill=NAVY)
    add_text(s, 7.30, 1.90, 5.3, 2.05, [
        [("$ python src/train_final.py", {"color": LBLUE})],
        [("    训练并保存模型", {"color": GRAY})],
        [("", {})],
        [("$ python src/predict.py face.jpg", {"color": WHITE})],
        [("    单张预测", {"color": GRAY})],
        [("", {})],
        [("$ python src/predict.py data/raw/down -r -q", {"color": WHITE})],
        [("    批量预测", {"color": GRAY})],
    ], size=10.5, spacing=1.30, space_after=0, font="Consolas")

    add_text(s, MARGIN, 4.28, 12.29, 0.35, "三个工程决定", size=15,
             color=NAVY, bold=True)

    cards = []
    for i, (t, d, c) in enumerate([
        ("共享特征模块", "训练与推理物理上共用同一份代码。描述子对对齐极敏感"
                        "（检测框平移 5px → 数值变化 0.043），两份代码漂移会导致"
                        "静默出错。", BLUE),
        ("加入拒识机制", "置信度低于阈值时输出「不确定」而非强行分类——"
                        "模型只见过 3 个病种，遇到第四种病仍会给高置信度错答。", RED),
        ("scaler 随模型存盘", "SVM 对特征尺度敏感。把标准化器与特征列序"
                             "一起存进 .joblib，防止推理时用错参数。", GREEN),
    ]):
        x = MARGIN + i * 4.13
        b = add_rect(s, x, 4.70, 3.95, 2.00, fill=WHITE, line=c, line_w=1.5,
                     shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
        add_text(s, x + 0.22, 4.86, 3.5, 0.32, t, size=12.5, color=c, bold=True)
        add_text(s, x + 0.22, 5.24, 3.5, 1.32, d, size=10, color=DGRAY,
                 spacing=1.34)
        cards.append(b)

    animate(s, [
        fade_all(tree, effect="wipe_right", dur=550),
        fade_all(code, effect="wipe_left", dur=550),
        [(c, "fade", i * 130, 420) for i, c in enumerate(cards)],
    ])
    return s


# ======================================================================
# P15 模型效果与能力边界
# ======================================================================
def p15_perf(prs):
    s = content_slide(prs, "模型效果与能力边界", 15, kicker="02 · 评估")

    # 三个指标卡
    metrics = [("AUC", "0.942"), ("准确率", "0.812"), ("macro F1", "0.809")]
    cards = []
    for i, (name, val) in enumerate(metrics):
        x = MARGIN + i * 2.0
        b = add_rect(s, x, 1.26, 1.85, 1.10, fill=NAVY,
                     shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.10)
        add_text(s, x + 0.12, 1.36, 1.61, 0.62, val, size=25, color=WHITE,
                 bold=True, align=CENTER, font=FONT_NUM)
        add_text(s, x + 0.12, 1.98, 1.61, 0.30, name, size=11, color=LBLUE,
                 align=CENTER)
        cards.append(b)

    conf = add_image_fit(s, ASSETS / "confusion.png", MARGIN, 2.52, 5.60, 3.10)

    thr = add_image_fit(s, ASSETS / "threshold_tradeoff.png", 6.32, 2.52,
                        3.16, 3.10)

    bounds = add_rect(s, 9.60, 2.52, 3.21, 3.10, fill=WHITE, line=RED,
                      line_w=1.6, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
                      radius=0.06)
    add_text(s, 9.82, 2.66, 2.80, 2.85, [
        [("⚠ 能力边界", {"size": 12.5, "bold": True, "color": RED})],
        [("仅支持 3 个病种，遇到第四种病或", {"size": 10})],
        [("健康人脸仍会输出高置信度结果", {"size": 10})],
        [("仅适用于接近正面的人脸", {"size": 10})],
        [("（正面 97% vs 侧脸 13% 检出率）", {"size": 10, "color": GRAY})],
        [("样本量仅 117，且无独立测试集", {"size": 10, "color": RED, "bold": True})],
        [("不可用于任何临床诊断", {"size": 10.5, "bold": True, "color": RED})],
    ], spacing=1.30, space_after=5)

    note = add_rect(s, MARGIN, 5.78, 12.29, 0.98, fill=BG, line=LGRAY,
                    shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
    add_text(s, MARGIN + 0.30, 5.90, 11.7, 0.78,
             [("口径说明：", {"bold": True, "color": NAVY}),
              ("0.942 是 10 折交叉验证对「这套流程」的估计，", {"color": DGRAY}),
              ("不是已保存模型自身的成绩", {"bold": True, "color": RED}),
              ("——它在全部 117 个样本上训练过，没有任何独立数据能衡量它。"
               "训练集内准确率 0.991 属无意义数字，不要引用。", {"color": DGRAY})],
             size=11.5, spacing=1.3)

    animate(s, [
        [(c, "wipe_up", i * 130, 420) for i, c in enumerate(cards)],
        fade_all(conf, effect="dissolve", dur=650),
        fade_all(thr, effect="wipe_right", dur=550),
        fade_all(bounds, effect="fade", dur=500),
        fade_all(note, effect="wipe_up", dur=450),
    ])
    return s


# ======================================================================
# P16 方案调整说明
# ======================================================================
def p16_adjust(prs):
    s = content_slide(prs, "方案调整说明（对照任务书）", 16, kicker="04 · 调整")

    hdr = [add_cell(s, MARGIN, 1.28, 2.06, 0.46, "项目", size=12.5,
                    color=WHITE, bold=True, fill=NAVY),
           add_cell(s, MARGIN + 2.06, 1.28, 2.20, 0.46, "任务书拟定", size=12.5,
                    color=WHITE, bold=True, fill=NAVY, align=CENTER),
           add_cell(s, MARGIN + 4.26, 1.28, 2.30, 0.46, "实际采用", size=12.5,
                    color=WHITE, bold=True, fill=NAVY, align=CENTER),
           add_cell(s, MARGIN + 6.56, 1.28, 5.73, 0.46, "调整理由", size=12.5,
                    color=WHITE, bold=True, fill=NAVY)]

    data = [
        ("分类模型", "XGBoost", "SVM",
         "实测 SVM AUC 0.94 > XGBoost 0.88，且 SVM 在深度/融合两组均为最优。"
         "建议按实测结果调整。", True),
        ("折划分", "分层 10 折", "分组分层 10 折",
         "同一受试者不跨折，避免信息泄漏，比论文更严格。", False),
        ("深度特征", "未指定模型", "Dlib ResNet 128 维",
         "论文采用方案，且无需额外训练。", False),
        ("模型解释", "SHAP", "未开始（计划中）",
         "安排在中期后。", False),
        ("后端", "Python+FastAPI（+SpringBoot）", "未开始",
         "需先确定 FastAPI 与 SpringBoot 的分工边界。", False),
    ]
    rows = []
    for i, (a, b, c, d, hi) in enumerate(data):
        y = 1.74 + i * 0.78
        fill = WHITE if hi else (PALE if i % 2 == 0 else WHITE)
        r = [add_cell(s, MARGIN, y, 2.06, 0.78, a, size=11.5, fill=fill,
                      bold=True, color=NAVY),
             add_cell(s, MARGIN + 2.06, y, 2.20, 0.78, b, size=10.5, fill=fill,
                      color=GRAY, align=CENTER),
             add_cell(s, MARGIN + 4.26, y, 2.30, 0.78, c, size=11, fill=fill,
                      color=RED if hi else GREEN, bold=True, align=CENTER),
             add_cell(s, MARGIN + 6.56, y, 5.73, 0.78, d, size=10.5, fill=fill)]
        rows.append(r)
        if hi:   # 高亮行加红框
            add_rect(s, MARGIN, y, 12.29, 0.78, fill=None, line=RED, line_w=1.8)

    dec = add_rect(s, MARGIN, 5.72, 12.29, 1.04, fill=BG, line=RED, line_w=1.5,
                   shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
    add_text(s, MARGIN + 0.32, 5.84, 11.7, 0.82, [
        [("🟡 待决策事项", {"size": 12.5, "bold": True, "color": RED})],
        [("① 分类模型是否改为 SVM？　② 后端是否引入 SpringBoot、"
          "如何与 FastAPI 分工？　③ 是否保留融合分支（建议不引入）？",
          {"size": 11, "color": DGRAY})],
    ], spacing=1.35, space_after=5)

    animate(s, [
        fade_all(*hdr, effect="fade", dur=350),
        [tuple(r) for r in rows],
        fade_all(dec, effect="wipe_up", dur=500),
    ])
    return s


# ======================================================================
# P17 困难与挑战
# ======================================================================
def p17_difficulty(prs):
    s = content_slide(prs, "当前困难与挑战", 17, kicker="04 · 困难")

    hdr = [add_cell(s, MARGIN, 1.28, 0.55, 0.46, "#", size=12,
                    color=WHITE, bold=True, fill=NAVY, align=CENTER),
           add_cell(s, MARGIN + 0.55, 1.28, 4.30, 0.46, "困难", size=12.5,
                    color=WHITE, bold=True, fill=NAVY),
           add_cell(s, MARGIN + 4.85, 1.28, 1.15, 0.46, "影响", size=12.5,
                    color=WHITE, bold=True, fill=NAVY, align=CENTER),
           add_cell(s, MARGIN + 6.00, 1.28, 6.29, 0.46, "应对措施", size=12.5,
                    color=WHITE, bold=True, fill=NAVY)]

    data = [
        ("1", "样本量极小（117），类别不均衡（53/33/31）", "高",
         "已用分组交叉验证 + 正则化；后续补充数据增强、迁移学习；不追求提高模型复杂度"),
        ("2", "无独立测试集，无法评估已保存模型的真实性能", "高",
         "计划从未参与训练的公开数据中划出外部验证集"),
        ("3", "输入条件苛刻（仅正面照，检出率 38%）", "中",
         "前端增加拍摄引导；考虑引入姿态估计与多角度人脸对齐"),
        ("4", "论文核心结论未复现（融合无增益）", "中",
         "已完成六组检验并给出解释；建议按实测结论调整技术方案（见 P16）"),
        ("5", "系统端尚未启动，工期压力集中在后半程", "中",
         "算法已封装为可调用模块，Web 端可直接对接，减少联调风险"),
        ("6", "SHAP 与区域可视化的特征映射问题", "低",
         "几何特征可直接映射到关键点区域；128 维深度特征需借助 Grad-CAM 类方法"),
    ]
    rows = []
    for i, (n, k, lvl, m) in enumerate(data):
        y = 1.74 + i * 0.79
        fill = PALE if i % 2 == 0 else WHITE
        lc = RED if lvl == "高" else (ORANGE if lvl == "中" else GRAY)
        r = [add_cell(s, MARGIN, y, 0.55, 0.79, n, size=11.5, fill=fill,
                      bold=True, color=NAVY, align=CENTER),
             add_cell(s, MARGIN + 0.55, y, 4.30, 0.79, k, size=11, fill=fill),
             add_cell(s, MARGIN + 4.85, y, 1.15, 0.79, lvl, size=11.5,
                      fill=fill, color=lc, bold=True, align=CENTER),
             add_cell(s, MARGIN + 6.00, y, 6.29, 0.79, m, size=10.5, fill=fill,
                      color=GREEN)]
        rows.append(r)

    animate(s, [
        fade_all(*hdr, effect="fade", dur=350),
        [tuple(r) for r in rows],
    ])
    return s


# ======================================================================
# P18 下一步工作计划
# ======================================================================
def p18_plan(prs):
    s = content_slide(prs, "下一步工作计划", 18, kicker="04 · 计划")

    g = add_image_fit(s, ASSETS / "gantt.png", MARGIN, 1.18, 12.29, 3.30)

    hdr = [add_cell(s, MARGIN, 4.62, 1.30, 0.42, "里程碑", size=12,
                    color=WHITE, bold=True, fill=NAVY, align=CENTER),
           add_cell(s, MARGIN + 1.30, 4.62, 1.30, 0.42, "时间", size=12,
                    color=WHITE, bold=True, fill=NAVY, align=CENTER),
           add_cell(s, MARGIN + 2.60, 4.62, 9.69, 0.42, "验收标准", size=12,
                    color=WHITE, bold=True, fill=NAVY)]
    ms = [
        ("M1", "第 11 周", "SHAP 解释模块跑通，能生成单样本特征贡献图"),
        ("M2", "第 13 周", "后端接口可用，支持图片上传 → 返回预测结果"),
        ("M3", "第 15 周", "前后端联调完成，系统可演示"),
        ("M4", "第 16 周", "完整系统交付 + 报告"),
    ]
    rows = []
    for i, (a, b, c) in enumerate(ms):
        y = 5.04 + i * 0.40
        fill = PALE if i % 2 == 0 else WHITE
        r = [add_cell(s, MARGIN, y, 1.30, 0.40, a, size=11, fill=fill,
                      bold=True, color=BLUE, align=CENTER),
             add_cell(s, MARGIN + 1.30, y, 1.30, 0.40, b, size=10.5, fill=fill,
                      color=GRAY, align=CENTER),
             add_cell(s, MARGIN + 2.60, y, 9.69, 0.40, c, size=10.5, fill=fill)]
        rows.append(r)

    tip = add_text(s, MARGIN, 6.70, 12.29, 0.32,
                   "关键路径是「后端 → 前端联调」；算法侧已无阻塞风险，"
                   "SHAP 与后端可并行推进。",
                   size=11.5, color=RED, align=CENTER, bold=True)

    animate(s, [
        fade_all(g, effect="dissolve", dur=750),
        fade_all(*hdr, effect="fade", dur=350),
        [tuple(r) for r in rows],
        fade_all(tip, effect="fade", dur=400),
    ])
    return s


# ======================================================================
# P19 分工与致谢
# ======================================================================
def p19_team(prs):
    s = content_slide(prs, "分工与致谢", 19, kicker="04 · 分工")

    add_text(s, MARGIN, 1.26, 12.29, 0.34, "组内分工", size=15,
             color=NAVY, bold=True)

    hdr = [add_cell(s, MARGIN, 1.66, 2.60, 0.44, "成员", size=12.5,
                    color=WHITE, bold=True, fill=NAVY),
           add_cell(s, MARGIN + 2.60, 1.66, 4.60, 0.44, "负责模块", size=12.5,
                    color=WHITE, bold=True, fill=NAVY),
           add_cell(s, MARGIN + 7.20, 1.66, 5.09, 0.44, "当前进展", size=12.5,
                    color=WHITE, bold=True, fill=NAVY)]

    names = ["梁顺健", "张锦霖", "王家柯", "蒋云浩"]
    rows = []
    for i, n in enumerate(names):
        y = 2.10 + i * 0.50
        fill = PALE if i % 2 == 0 else WHITE
        r = [add_cell(s, MARGIN, y, 2.60, 0.50, n, size=12, fill=fill,
                      bold=True, color=NAVY),
             add_cell(s, MARGIN + 2.60, y, 4.60, 0.50, "（待填写）", size=11,
                      fill=fill, color=RED),
             add_cell(s, MARGIN + 7.20, y, 5.09, 0.50, "（待填写）", size=11,
                      fill=fill, color=RED)]
        rows.append(r)

    hint = add_text(s, MARGIN, 4.16, 12.29, 0.32,
                    "⚠ 汇报前请按实际情况填写本表 —— 中期答辩必查项",
                    size=11.5, color=RED, bold=True)

    achieve = add_rect(s, MARGIN, 4.62, 6.05, 2.10, fill=BG, line=GREEN,
                       line_w=1.5, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
                       radius=0.08)
    add_text(s, MARGIN + 0.28, 4.76, 5.5, 1.85, [
        [("阶段性成果", {"size": 13, "bold": True, "color": GREEN})],
        [("✅ 完整复现论文 8 个步骤，产出 12 组对比实验", {"size": 10.5})],
        [("✅ 6 组独立检验，完成对论文核心结论的证伪", {"size": 10.5})],
        [("✅ 1 个可调用预测模型（CLI + Python API + 文档）", {"size": 10.5})],
        [("✅ 3 份技术文档 + 2065 行算法代码", {"size": 10.5})],
    ], spacing=1.34, space_after=5)

    thanks = add_rect(s, 6.76, 4.62, 6.05, 2.10, fill=NAVY)
    add_text(s, 6.76, 5.16, 6.05, 0.52, "恳请各位老师批评指正",
             size=22, color=WHITE, bold=True, align=CENTER)
    add_text(s, 6.76, 5.86, 6.05, 0.60, "Q & A", size=26, color=LBLUE,
             bold=True, align=CENTER, font=FONT_NUM)

    animate(s, [
        fade_all(*hdr, effect="fade", dur=350),
        [tuple(r) for r in rows],
        fade_all(hint, effect="fade", dur=400),
        fade_all(achieve, effect="wipe_right", dur=500),
        fade_all(thanks, effect="dissolve", dur=700),
    ])
    return s


# ======================================================================
def main():
    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)

    for fn in (p01_cover, p02_outline, p03_background, p04_taskbook,
               p05_pipeline, p06_progress, p07_dataset, p08_geo, p09_deep,
               p10_select, p11_design, p12_results, p13_tests, p14_deliver,
               p15_perf, p16_adjust, p17_difficulty, p18_plan, p19_team):
        fn(prs)
        print(f"  ✓ {fn.__name__}")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT_FILE)
    size = OUT_FILE.stat().st_size / 1024 / 1024
    print(f"\n已保存: {OUT_FILE.relative_to(ROOT)}  ({size:.2f} MB)")
    print(f"共 {len(prs.slides._sldIdLst)} 页，含切换与进入动画")


if __name__ == "__main__":
    main()
