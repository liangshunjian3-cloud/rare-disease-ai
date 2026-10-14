# -*- coding: utf-8 -*-
"""
构建 PPT 的底层工具箱：版式、形状、表格、图片，以及**动画注入**。

为什么要自己注入动画
--------------------
python-pptx 不支持动画（官方 issue #444，至今未实现）。PPT 的动画存在
slide XML 的 `<p:timing>` 节点里，只能手工拼 XML 再挂上去。本模块把这段
XML 的生成封装成 `animate(slide, steps)`，调用方只管给出「第几次单击、
哪些形状、什么效果」。

`<p:sld>` 的子元素顺序是**有 schema 约束**的：
    cSld → clrMapOvr → transition → timing → extLst
必须按这个顺序插入，否则 PowerPoint 会报「需要修复」。python-pptx 建出来的
空 slide 自带 cSld 与 clrMapOvr，所以这里只需依次 append transition 和
timing 即可，顺序天然正确。
"""

from pathlib import Path

from lxml import etree
from PIL import Image
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

# ======================================================================
# 视觉规范（与 docs/MIDTERM_PPT.md 一致）
# ======================================================================
NAVY = RGBColor(0x1F, 0x38, 0x64)
BLUE = RGBColor(0x2E, 0x75, 0xB6)
LBLUE = RGBColor(0x9D, 0xC3, 0xE6)
PALE = RGBColor(0xEC, 0xF2, 0xFA)
RED = RGBColor(0xC0, 0x00, 0x00)
GREEN = RGBColor(0x54, 0x82, 0x35)
ORANGE = RGBColor(0xED, 0x7D, 0x31)
GRAY = RGBColor(0x7F, 0x7F, 0x7F)
DGRAY = RGBColor(0x40, 0x40, 0x40)
LGRAY = RGBColor(0xD9, 0xD9, 0xD9)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BG = RGBColor(0xF7, 0xF9, 0xFC)

FONT = "微软雅黑"
FONT_NUM = "Arial"

SLIDE_W = 13.333
SLIDE_H = 7.5
MARGIN = 0.52
HEADER_H = 0.95
BODY_TOP = 1.22
FOOTER_Y = 7.02


# ======================================================================
# 基础形状
# ======================================================================
def _set_font(run, name=FONT):
    """同时设置拉丁与东亚字体，避免中文回退成宋体。"""
    run.font.name = name
    rPr = run._r.get_or_add_rPr()
    for tag in ("a:latin", "a:ea", "a:cs"):
        el = rPr.find(qn(tag))
        if el is None:
            el = etree.SubElement(rPr, qn(tag))
        el.set("typeface", name)


def add_rect(slide, l, t, w, h, fill=None, line=None, line_w=1.0,
             shape=MSO_SHAPE.RECTANGLE, radius=None, shadow=False):
    shp = slide.shapes.add_shape(shape, Inches(l), Inches(t), Inches(w), Inches(h))
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(line_w)
    if radius is not None and shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        shp.adjustments[0] = radius
    if not shadow:
        # 关掉默认阴影，否则每个形状下面都有一层灰影
        spPr = shp._element.spPr
        el = etree.SubElement(spPr, qn("a:effectLst"))
    shp.text_frame.word_wrap = True
    return shp


def add_text(slide, l, t, w, h, runs, size=18, color=DGRAY, bold=False,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spacing=1.18,
             space_after=4, font=FONT, italic=False):
    """runs 支持三种写法：

        "一行文字"                        单段单run
        [("粗", {"bold": True}), "普通"]   单段多run（同段内混排）
        [[...], [...]]                    多段，每个子列表是一段

    文本里的 "\\n" 会转成真正的软换行（<a:br/>），而不是塞进 <a:t> 当空白。
    """
    box = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Emu(0)
    tf.margin_top = tf.margin_bottom = Emu(0)

    # 消歧：列表首元素是 (文本, 属性) 元组 → 视为单段内的多 run
    if isinstance(runs, str):
        paras = [runs]
    elif runs and isinstance(runs[0], tuple):
        paras = [runs]
    else:
        paras = runs

    for i, item in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        p.space_after = Pt(space_after)

        parts = item if isinstance(item, list) else [(item, {})]
        for text, ov in parts:
            segs = str(text).split("\n")
            for si, seg in enumerate(segs):
                if si > 0:
                    p.add_line_break()
                if not seg and len(segs) > 1:
                    continue
                r = p.add_run()
                r.text = seg
                r.font.size = Pt(ov.get("size", size))
                r.font.bold = ov.get("bold", bold)
                r.font.italic = ov.get("italic", italic)
                r.font.color.rgb = ov.get("color", color)
                _set_font(r, ov.get("font", font))

    return box


def add_image_fit(slide, path, l, t, w, h, max_scale=1.0):
    """按比例缩放并居中放进 (l,t,w,h) 框里。"""
    img_w, img_h = Image.open(path).size
    scale = min(w / img_w, h / img_h) * max_scale
    nw, nh = img_w * scale, img_h * scale
    left = l + (w - nw) / 2
    top = t + (h - nh) / 2
    return slide.shapes.add_picture(str(path), Inches(left), Inches(top),
                                    Inches(nw), Inches(nh))


def add_cell(slide, l, t, w, h, text, size=13, color=DGRAY, bold=False,
             fill=WHITE, align=PP_ALIGN.LEFT, pad=0.10):
    """表格单元格 = 一个带文字的矩形。用形状而非原生表格，方便逐行动画。"""
    shp = add_rect(slide, l, t, w, h, fill=fill, line=LGRAY, line_w=0.75)
    tf = shp.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(pad)
    tf.margin_right = Inches(pad)
    tf.margin_top = tf.margin_bottom = Emu(0)
    p = tf.paragraphs[0]
    p.alignment = align
    p.line_spacing = 1.0
    r = p.add_run()
    r.text = str(text)
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    _set_font(r)
    return shp


def add_table(slide, l, t, col_widths, headers, rows, row_h=0.42,
              head_h=0.46, font_size=12.5, head_size=12.5,
              shade=None, aligns=None, colors=None):
    """用单元格矩形拼一张表。

    shade  : {'行号': 填充色} 或 None（默认表头深蓝、正文斑马纹）
    colors : {'行号': 文字色}
    aligns : 每列的对齐方式
    返回 [[每行的形状], ...]，第 0 行是表头。
    """
    n_col = len(col_widths)
    aligns = aligns or [PP_ALIGN.LEFT] * n_col
    colors = colors or {}
    total_w = sum(col_widths)

    def col_x(ci):
        return l + sum(col_widths[:ci])

    out = []

    # 表头
    head = []
    for ci, htext in enumerate(headers):
        head.append(add_cell(slide, col_x(ci), t, col_widths[ci], head_h,
                             htext, size=head_size, color=WHITE, bold=True,
                             fill=NAVY, align=aligns[ci]))
    out.append(head)

    # 正文
    for ri, row in enumerate(rows):
        y = t + head_h + ri * row_h
        if shade and ri in shade:
            fill = shade[ri]
        else:
            fill = PALE if ri % 2 == 0 else WHITE
        cellcolor = colors.get(ri, DGRAY)
        cells = []
        for ci, val in enumerate(row):
            cells.append(add_cell(slide, col_x(ci), y, col_widths[ci], row_h,
                                  val, size=font_size, color=cellcolor,
                                  fill=fill, align=aligns[ci]))
        out.append(cells)

    return out


# ======================================================================
# 页面骨架
# ======================================================================
def add_header(slide, title, kicker=None):
    """顶部深蓝标题条（不参与动画，幻灯片一出现就在）。"""
    add_rect(slide, 0, 0, SLIDE_W, HEADER_H, fill=NAVY)
    add_text(slide, MARGIN, 0.17, 10.4, 0.42, title, size=26, color=WHITE,
             bold=True, anchor=MSO_ANCHOR.MIDDLE)
    if kicker:
        add_text(slide, SLIDE_W - MARGIN - 3.6, 0.26, 3.6, 0.36, kicker,
                 size=12, color=LBLUE, align=PP_ALIGN.RIGHT,
                 anchor=MSO_ANCHOR.MIDDLE)


def add_footer(slide, page):
    add_text(slide, MARGIN, FOOTER_Y, 5.0, 0.3,
             "哈工大（深圳） · 组号 3", size=10, color=GRAY)
    add_text(slide, SLIDE_W - MARGIN - 1.2, FOOTER_Y, 1.2, 0.3,
             str(page), size=11, color=GRAY, align=PP_ALIGN.RIGHT, font=FONT_NUM)


def content_slide(prs, title, page, kicker=None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_rect(slide, 0, 0, SLIDE_W, SLIDE_H, fill=WHITE)
    add_header(slide, title, kicker)
    add_footer(slide, page)
    return slide


# ======================================================================
# 动画注入
# ======================================================================
# presetID 取自 PowerPoint 内置的进入动画编号
EFFECTS = {
    "fade":      (10, "fade"),
    "wipe_up":   (22, "wipe(up)"),
    "wipe_left": (22, "wipe(left)"),
    "wipe_right": (22, "wipe(right)"),
    "dissolve":  (9, "dissolve"),
}


def _norm(item, default_eff="fade", default_dur=450):
    """把一步里的元素统一成 (目标, 效果, 延迟, 时长)。

    目标可以是单个形状，也可以是一组形状（同时出现）。
    判定依据：四元组且第 2 个元素是字符串（效果名）才算显式参数，
    否则整项就是一个形状或一组形状。
    """
    if (isinstance(item, tuple) and len(item) == 4
            and isinstance(item[1], str)):
        return item
    return (item, default_eff, 0, default_dur)


def _iter_targets(target):
    if isinstance(target, (list, tuple)):
        return list(target)
    return [target]


def _timing_xml(steps):
    """steps: [[元素, ...], ...]，每个子列表 = 一次单击（之后各步再单击）。

    元素可以是：
        形状                            默认 fade
        (形状, 效果, 延迟ms, 时长ms)      显式参数
        (形状组, 效果, 延迟ms, 时长ms)    一组同时出现
        形状组                          默认 fade，同时出现
    """
    cid = [1]

    def nid():
        cid[0] += 1
        return cid[0]

    click_pars = []
    for step in steps:
        # 先把这一步展开成扁平的效果列表，再按序号定 nodeType
        flat = []
        for item in step:
            target, eff, delay, dur = _norm(item)
            for shp in _iter_targets(target):
                flat.append((shp, eff, delay, dur))

        effects = []
        for i, (shape, eff, delay, dur) in enumerate(flat):
            preset, filt = EFFECTS[eff]
            spid = shape.shape_id
            # 一次单击内的第一个是 clickEffect，其余是 withEffect
            node = "clickEffect" if i == 0 else "withEffect"
            effects.append(f'''
                    <p:par>
                      <p:cTn id="{nid()}" presetID="{preset}" presetClass="entr"
                             presetSubtype="0" fill="hold" grpId="0" nodeType="{node}">
                        <p:stCondLst><p:cond delay="{delay}"/></p:stCondLst>
                        <p:childTnLst>
                          <p:set>
                            <p:cBhvr>
                              <p:cTn id="{nid()}" dur="1" fill="hold">
                                <p:stCondLst><p:cond delay="0"/></p:stCondLst>
                              </p:cTn>
                              <p:tgtEl><p:spTgt spid="{spid}"/></p:tgtEl>
                              <p:attrNameLst><p:attrName>style.visibility</p:attrName></p:attrNameLst>
                            </p:cBhvr>
                            <p:to><p:strVal val="visible"/></p:to>
                          </p:set>
                          <p:animEffect transition="in" filter="{filt}">
                            <p:cBhvr>
                              <p:cTn id="{nid()}" dur="{dur}"/>
                              <p:tgtEl><p:spTgt spid="{spid}"/></p:tgtEl>
                            </p:cBhvr>
                          </p:animEffect>
                        </p:childTnLst>
                      </p:cTn>
                    </p:par>''')

        click_pars.append(f'''
              <p:par>
                <p:cTn id="{nid()}" fill="hold">
                  <p:stCondLst><p:cond delay="indefinite"/></p:stCondLst>
                  <p:childTnLst>
                    <p:par>
                      <p:cTn id="{nid()}" fill="hold">
                        <p:stCondLst><p:cond delay="0"/></p:stCondLst>
                        <p:childTnLst>{"".join(effects)}
                        </p:childTnLst>
                      </p:cTn>
                    </p:par>
                  </p:childTnLst>
                </p:cTn>
              </p:par>''')

    return f'''<p:timing xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
           xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
      <p:tnLst>
        <p:par>
          <p:cTn id="1" dur="indefinite" restart="never" nodeType="tmRoot">
            <p:childTnLst>
              <p:seq concurrent="1" nextAc="seek">
                <p:cTn id="2" dur="indefinite" nodeType="mainSeq">
                  <p:childTnLst>{"".join(click_pars)}
                  </p:childTnLst>
                </p:cTn>
                <p:prevCondLst>
                  <p:cond evt="onPrev" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond>
                </p:prevCondLst>
                <p:nextCondLst>
                  <p:cond evt="onNext" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond>
                </p:nextCondLst>
              </p:seq>
            </p:childTnLst>
          </p:cTn>
        </p:par>
      </p:tnLst>
    </p:timing>'''


TRANSITION_XML = '''<p:transition xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
       xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
       spd="med" advClick="1">
  <p:fade/>
</p:transition>'''


def set_transition(slide, kind="fade"):
    """页面切换。必须插在 clrMapOvr 之后、timing 之前。"""
    sld = slide._element
    for old in sld.findall(qn("p:transition")):
        sld.remove(old)
    el = etree.fromstring(TRANSITION_XML.replace("<p:fade/>",
                                                 f"<p:{kind}/>"))
    # 插到 clrMapOvr 之后（若无则插到 cSld 之后）
    anchor = sld.find(qn("p:clrMapOvr"))
    if anchor is None:
        anchor = sld.find(qn("p:cSld"))
    anchor.addnext(el)


def animate(slide, steps, transition="fade"):
    """给 slide 挂上进入动画。

    steps: [[(形状, 效果, 延迟ms, 时长ms), ...], ...]
           每个子列表对应「一次单击」，表内元素同时（或按延迟错开）出现。
    """
    sld = slide._element
    for old in sld.findall(qn("p:timing")):
        sld.remove(old)

    set_transition(slide, transition)
    if not steps:
        return
    sld.append(etree.fromstring(_timing_xml(steps)))


def fade_all(*shapes, delay=0, dur=450, effect="fade"):
    """把若干形状打包成「一次单击、一起淡入」的便利写法。"""
    return [(s, effect, delay, dur) for s in shapes]
