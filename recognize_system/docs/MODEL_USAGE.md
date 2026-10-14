# 模型使用简介

本文档说明如何使用本项目训练出的罕见病面部识别模型。

---

## ⚠️ 先读这一段

**本模型仅用于课程作业与算法复现，不可用于任何临床诊断、筛查或医疗决策。**

理由是硬性的，不是客套：

| 问题 | 现状 |
|---|---|
| 样本量 | 117 张图 / 109 位受试者 / 3 个病种 |
| 独立测试集 | **没有**。所有样本都参与了训练 |
| 可识别的病种 | 只有 DiGeorge、唐氏、Williams 三种；遇到第四种病仍会强行归到这三类里 |
| 数据来源 | 公开数据集的人脸照片，非临床采集，无人口学与病程信息 |

主预测脚本会把 128 维深度特征流程的 10 折交叉验证结果写入主模型文件；
几何辅助模型也单独保存其交叉验证参考指标。这些都不是独立测试成绩
（细节见 [§7](#7-模型文件里存了什么)）。
要真正评估它，必须另找一批从未参与训练的数据。

---

## 1. 环境准备

### 1.1 依赖

```bash
pip install -r requirements.txt
```

本机验证通过的版本：Python 3.13.5、dlib 20.0.1、scikit-learn 1.7.1、
numpy 2.3.2、pandas 2.3.2、Pillow 11.3.0、joblib 1.5.2。

按需启用 SHAP 解释时还需安装：

```bash
pip install shap
```

> **dlib 单独说明**：它是 C++ 扩展，`pip install dlib` 需要本机有 CMake 与 C++ 编译器。
> Windows 上最省事的方式是装预编译 wheel，或改用 conda：`conda install -c conda-forge dlib`。

### 1.2 模型文件

`models/` 下需要两个 dlib 模型来运行主预测；几何辅助模型只在使用 `--explain` 时需要：

```
models/
├── shape_predictor_68_face_landmarks.dat              68 点关键点
├── dlib_face_recognition_resnet_model_v1.dat          128 维主预测描述子模型
└── final/
    ├── svm_deep.joblib                               128 维深度主分类器
    ├── svm_geo61.joblib                              61 维几何辅助分类器与 SHAP 背景样本
    └── threshold_table.csv                                阈值对照表
```

主预测使用 128 维 dlib 描述子。启用 SHAP 时，系统会另用原图坐标系的 68 点重建
61 维语义几何特征，并由独立的几何辅助分类器解释主模型给出的类别。该 SHAP
结果不是深度主模型的直接归因，也不解释图像像素区域。若 dlib 模型缺失，运行时会明确报错提示下载地址：

```bash
# 手动下载（Linux/macOS）
cd models
curl -O http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
curl -O http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2
bunzip2 dlib_face_recognition_resnet_model_v1.dat.bz2
```

`svm_deep.joblib` 不在？确保已有 `data/embeddings_128.csv` 后运行
`python src/train_final.py` 生成。`svm_geo61.joblib` 不在？确保已有
`data/landmarks_68_raw.csv` 后运行 `python src/train_geometry.py` 生成。

---

## 2. 训练、预测与解释

```bash
# ① 训练主预测模型（已训练过可跳过）
python src/train_final.py

# ② （可选）训练用于解释的几何辅助模型
python src/train_geometry.py

# ③ 预测一张图
python src/predict.py 你的图片.jpg

# ④ 批量预测一个目录
python src/predict.py data/raw/down -r -q

# ⑤ 解释主模型最高概率类别对应的几何辅助模型特征
python src/predict.py 你的图片.jpg --explain
```

输出长这样：

```
图片: data\raw\digeorge\digeorge-argentina-patient-14\01.jpg

  预测病种 : DiGeorge 综合征（DiGeorge）
  置信度   : 0.99

  各类概率 :
    DiGeorge 综合征     0.993  ████████████
    唐氏综合征            0.000
    Williams 综合征     0.006
```

> **Windows 用户注意**：若控制台出现 `UnicodeEncodeError`，在命令前加
> `PYTHONIOENCODING=utf-8`（Git Bash / PowerShell 均可）。

---

## 3. 命令行完整用法

```
python src/predict.py <路径> [选项]
```

| 选项 | 说明 |
|---|---|
| `路径` | 单张图片，或一个目录（目录必须搭配 `-r`） |
| `-r, --recursive` | 递归处理目录下所有图片 |
| `-q, --quiet` | 批量时只打印汇总表，不逐张展开 |
| `-t, --threshold` | 拒识阈值，范围 0~1。越大越保守，默认 `0.6` |
| `-m, --model` | 指定深度主模型文件，默认 `models/final/svm_deep.joblib` |
| `--geometry-model` | 指定几何辅助模型文件，默认 `models/final/svm_geo61.joblib` |
| `--json` | 输出 JSON，便于被其他程序调用 |
| `--explain` | 用 SHAP 输出几何辅助模型对主预测类别的前 10 个几何特征贡献（需安装 `shap`，计算较慢） |

支持的图片格式：`.jpg .jpeg .png .bmp .webp`

### 退出码

写脚本调用时按这个判断，不要靠解析文字：

| 码 | 含义 |
|---|---|
| `0` | 全部成功判定 |
| `1` | 有图片处理失败（未检出人脸、文件损坏等） |
| `2` | 参数或路径错误、模型文件缺失 |

### 常用例子

```bash
# 单独调高阈值，更保守
python src/predict.py photo.jpg -t 0.8

# 递归批量，只看汇总
python src/predict.py data/raw/williams -r -q

# 输出 JSON 供程序消费
python src/predict.py photo.jpg --json
```

`--json` 的输出结构：

```json
{
  "model":   { "cv_metrics": {...}, "n_samples": 117, "trained_at": "..." },
  "prediction_model": "128维 dlib 深度分类器",
  "explanation_model": null,
  "threshold": 0.6,
  "results": [
    {
      "image_path": "photo.jpg",
      "label": "DiGeorge",
      "label_cn": "DiGeorge 综合征",
      "confidence": 0.993,
      "probs": { "DiGeorge": 0.993, "Down": 0.0004, "Williams": 0.006 },
      "rejected": false,
      "threshold": 0.6,
      "explanation": null,
      "error": null
    }
  ]
}
```

---

## 4. Python 接口

```python
from src.predict import load_model, predict_image, predict_batch, collect_images

# dlib 模型加载较慢，批量推理务必复用同一个 bundle
bundle = load_model()

r = predict_image("photo.jpg", bundle)

if r.error:                 # 没检出人脸 / 文件损坏
    print("失败:", r.error)
elif r.rejected:            # 置信度不足，模型主动不表态
    print(f"不确定（最高仅 {r.confidence:.2f}）")
else:
    print(r.label_cn, f"{r.confidence:.2f}")   # 唐氏综合征 0.98
    print(r.probs)                             # {'DiGeorge': ..., 'Down': ..., 'Williams': ...}

# 按需计算 SHAP（解释几何辅助模型对主模型最高概率类别的几何特征贡献）
# 解释器是独立的辅助模型，不是深度主模型的直接 SHAP 归因。
r = predict_image("photo.jpg", bundle, explain=True)
print(r.explanation)

# 批量：自己收集路径，一次加载模型跑完
paths = collect_images("data/raw/williams", recursive=True)
for r in predict_batch(paths, bundle, threshold=0.7):
    print(r.image_path, r.label_cn if r.ok else "(未判定)")
```

### `Prediction` 对象字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `image_path` | str | 输入路径 |
| `label` / `label_cn` | str \| None | 病种英文名 / 中文名；拒识或出错时为 `None` |
| `confidence` | float | 最高类别的概率 |
| `probs` | dict | 三个类别各自的概率，和为 1 |
| `rejected` | bool | 是否因置信度低于阈值而未表态 |
| `threshold` | float \| None | **本次判定实际使用**的阈值 |
| `explanation` | dict \| None | 可选的几何辅助模型 SHAP 局部解释；不是深度主模型的直接归因 |
| `error` | str \| None | 处理失败的原因；非 `None` 表示没走到模型 |
| `.ok` | property | `error is None and not rejected and label is not None` |

三种状态互斥，判断时优先看 `error`，再看 `rejected`：

```
error 非空      → 这张图没处理成功（通常是没检出人脸）
rejected=True   → 处理成功了，但模型没把握，不输出病种
ok=True         → 给出了明确的病种判断
```

## 4.1 FastAPI 前端接口

前端服务复用本节的 `load_model()` 和 `predict_image()`，不会另行实现一套预测逻辑。
接口默认执行与 `--explain` 相同的几何辅助模型 SHAP 解释；SHAP 计算可能较慢。
每次解释最多返回正向 SHAP 贡献最高的五个几何特征；若正向贡献不足五项，则只返回实际存在的正向项。
模型和 dlib 人脸组件按服务进程惰性加载一次，后续请求复用已加载的模型。

从项目根目录启动服务：

```powershell
pip install -r requirements_ascii.txt
python -m uvicorn main:app --app-dir ai_service --host 127.0.0.1 --port 8000
```

确保 `recognize_system/models/final/svm_deep.joblib`、`svm_geo61.joblib` 和
dlib 的两个 `.dat` 模型文件都已就位。SHAP、FastAPI、Uvicorn 及 multipart 表单依赖
已列在 `requirements_ascii.txt` 中。`/health` 可用于检查服务是否启动；交互式 API
文档位于 `http://127.0.0.1:8000/docs`。

### `POST /predict`

以 `multipart/form-data` 上传一张 `file`，支持 JPG、PNG、WebP，最大 10 MB。
可选表单字段 `explain` 默认为 `true`；传 `false` 可跳过 SHAP 解释。可选字段
`threshold` 是 0 到 1 之间的拒识阈值，省略时使用主模型默认值。前端 Vite 开发代理
已将 `/api/predict` 转发到该接口的 `/predict`。

成功时返回 JSON。`topK` 包含按深度主模型概率降序排列的所有类别，`probabilities`
是按英文类别名索引的同一组概率；`explanations` 仅包含几何辅助模型 SHAP 值大于零的特征，
按正向贡献从高到低最多返回五项；若不足五项则不以负向特征补齐。正数 `shapValue`
表示相对 SHAP 基准值推高被解释类别概率。

```json
{
  "status": "ok",
  "prediction": {
    "label": "唐氏综合征",
    "labelEn": "Down",
    "probability": 0.87,
    "rejected": false,
    "threshold": 0.6
  },
  "topK": [
    {"label": "唐氏综合征", "labelEn": "Down", "probability": 0.87},
    {"label": "DiGeorge 综合征", "labelEn": "DiGeorge", "probability": 0.08},
    {"label": "Williams 综合征", "labelEn": "Williams", "probability": 0.05}
  ],
  "probabilities": {"DiGeorge": 0.08, "Down": 0.87, "Williams": 0.05},
  "explanation": {
    "class": "Down",
    "classCn": "唐氏综合征",
    "model": "61维几何辅助分类器",
    "geometryModelPrediction": "Down",
    "geometryModelPredictionCn": "唐氏综合征",
    "geometryModelPredictionProbability": 0.71,
    "baseValue": 0.34,
    "predictedProbability": 0.71
  },
  "explanations": [
    {
      "name": "眼间距",
      "feature": "eyes_dist",
      "value": 0.2,
      "shapValue": 0.12,
      "effect": "推高该类概率"
    }
  ],
  "warning": "概率仅表示三个候选类别之间的模型输出……"
}
```

上面数值仅用于说明响应结构，不是模型的实际输出。若最高概率低于阈值，仍会返回
各类深度模型概率及几何解释，但 `status` 为 `rejected`、`prediction.label` 为「不确定」。
上传类型错误返回 400，超过大小限制返回 413，无法处理图像（例如未检测到人脸）返回
422，模型文件或解释依赖不可用返回 503。

**概率是候选类别之间的模型分数，不代表患病概率。SHAP 特征由独立的几何辅助模型计算，
不是 128 维深度主模型的直接归因，也不是因果或临床证据。本服务仅用于课程作业与算法复现，
不可用于诊断、筛查或医疗决策。**

---

## 5. 怎么理解输出

### `confidence` 不是「患病概率」

它是 SVC 概率校准得到的**最高类别相对概率**，只在三个候选病种之间做比较，回答的是
「这张脸更像三者中的哪一个」。它**不能**回答「这个人有多大可能是患者」——
因为训练集里 100% 都是患者，模型从未见过健康人脸，也就无从学出这个界限。

这也是拒识机制存在的原因：给一张不属于这三类的脸，模型照样会输出一个很高的
`confidence`，只是它没有意义。阈值能挡掉一部分，但挡不干净。

### 拒识阈值 `-t` 怎么选

默认值为 `0.6`。主模型训练脚本会基于 128 维深度模型的 out-of-fold 概率重新生成
`models/final/threshold_table.csv`。部署前应检查该表，并据实际需求选择阈值。

**怎么选**：宁可漏判也不要误判的场景调高（0.7~0.8）；只看大概分布、不介意噪声的
场景用默认或更低。这张表是在 117 个样本上统计出来的，样本量小，别把小数点后
两位当回事。

### 各类别表现

主预测使用 128 维深度模型；几何辅助模型仅用于输出可读的结构特征解释。
重训后请依据主模型的 out-of-fold 指标评估；现阶段没有独立测试集。

---

## 6. 能力边界（重要）

### 只接受接近正面的脸

这是**实测**的，不是推测。对数据集 330 张图按文件名统计检出率：

| 文件名 | 检出 / 总数 | 检出率 |
|---|---|---|
| `01.jpg` | 110 / 113 | **97%** |
| `02.jpg` | 12 / 95 | 13% |
| `03.jpg` | 2 / 60 | 3% |
| `04.jpg` | 2 / 42 | 5% |
| `05.jpg` | 1 / 14 | 7% |
| `06`~`08.jpg` | 0 / 6 | 0% |
| **合计** | **127 / 330** | **38%** |

数据集约定 `01.jpg` 是正面照，`02` 及以后是侧脸、仰角、眼部特写等。
我们目视确认过一张 `02.jpg`：是孩子的 90° 侧面照，检不出人脸是**正确行为**。

> 所以：**330 张里只有 127 张能进模型，这不是 bug，是这套流程的适用范围。**
> 之前有人（包括我）把剩下的 203 张当成「可以用来做数据增强的闲置数据」——
> 这是错的，它们绝大多数是本流程处理不了的侧脸。

**实际使用建议**：输入要拍成正面、正对镜头、光照均匀、人脸占据画面较大比例的
照片。稍微歪头（±15° 内）通常可以，明显的侧脸、低头、仰头基本会直接报
「未检测到人脸」。

### 其他边界

- **只有 3 个病种**。输入其他罕见病或健康人脸时，模型会给一个高置信度的错误答案。
  拒识阈值能挡掉一部分，不能指望它。
- **不做人脸识别**。模型输出的是「像哪类病」，不是「这是谁」。同一受试者的不同
  照片会被判成同一病种，但模型不知道他们是同一个人。
- **未见过的种族/年龄段**表现未知。训练数据以公开数据集为主，覆盖的族群有限。
- **多张人脸只取最大的那张**。群照里其他人会被忽略。

---

## 7. 模型文件里存了什么

主预测文件 `models/final/svm_deep.joblib` 与辅助解释文件
`models/final/svm_geo61.joblib` 都是 dict（用 `joblib.load` 读）。

主模型包含：

| 键 | 内容 |
|---|---|
| `model` | `SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)` |
| `scaler` | `StandardScaler`，推理前必须用它变换特征 |
| `feature_cols` | 128 个 dlib 描述子列名，**顺序必须与推理时一致** |
| `label_map` | `{"digeorge": 0, "down": 1, "williams": 2}` |
| `label_names` / `label_names_cn` | 类别英文名 / 中文名 |
| `default_threshold` | 默认拒识阈值 `0.6` |
| `cv_metrics` | 交叉验证估计：`auc_micro` / `accuracy` / `f1_macro` |
| `cv_protocol` | 折划分方式说明 |
| `n_samples` / `n_subjects` | 117 / 109 |
| `trained_at` / `seed` | 训练时间 / 随机种子 42 |

几何辅助模型包含相同的 `model`、`scaler` 和类别信息，以及 `feature_cols`
（61 个语义特征名）和 `shap_background`（按类别均衡抽取的训练几何特征）。
它不参与主预测，也不决定拒识阈值。

**为什么 scaler 要跟模型一起存**：SVM 对特征尺度敏感，推理时若用了不同的
标准化参数（或忘记标准化），预测会**静默出错**——不报错，但结果是错的。
把 scaler 塞进同一个文件是防止这件事最直接的办法。

**为什么特征列顺序重要**：深度描述子由 dlib 固定生成 128 维；几何特征向量则按
`extract_features_61.py` 定义顺序生成。列序错位等于把特征打乱喂给模型，不要手工修改
bundle 中的特征名或顺序。

**怎么理解 SHAP**：每项贡献解释的是相对于背景样本的类别概率变化。正值表示推高被解释
类别的几何辅助模型概率，负值表示压低；这不是因果关系，也不是临床证据。SHAP
只解释几何辅助分类器，不能视作深度主模型的忠实归因，也不解释图像像素或 128 维嵌入。

---

## 8. 重新训练

```bash
python src/train_final.py              # 默认阈值 0.6
python src/train_final.py --threshold 0.7
python src/train_geometry.py           # 训练 61 维几何解释辅助模型
```

`train_final.py` 训练 128 维深度主模型并做以下四件事：

1. 读 `data/embeddings_128.csv` 中提取好的 128 维 dlib 描述子
2. 跑 10 折交叉验证（`StratifiedGroupKFold`，**按受试者分组**，同一人不跨折）
   —— 结果只作为「这套流程」的参考，不代表最终模型
3. 基于 out-of-fold 概率生成阈值对照表
4. 在**全部 117 个样本**上训练并保存

第 2 步和第 4 步的口径差异是刻意分开的，脚本运行时也会打印提醒：

实际指标以重训时打印和保存的结果为准。
交叉验证里的每个折模型都只在部分数据上训练过、训练完就丢了，和保存下来的最终模型不是同一个东西。

**几何辅助模型特征来源**：`train_geometry.py` 与在线解释都从原图坐标系的 68 个关键点
提取同样的 61 维特征；不要用裁剪图坐标系的 `landmarks_68.csv` 训练辅助模型。
单独运行 `python src/train_geometry.py` 会训练辅助模型，不会修改主预测模型或其拒识阈值表。

---

## 9. 常见问题

**`未检测到人脸（上采样 1 次）`**
最常见的情况，通常是侧脸、人脸太小、光照过暗或过度曝光。先确认是正面清晰照，
再考虑调大 `src/face_embedding.py` 里的 `DETECT_UPSAMPLE`（代价是耗时按平方增长）。

改动 `DETECT_UPSAMPLE` 会改变检测框，可能影响关键点和几何特征；修改后应重新提取关键点、重新训练几何辅助模型并重新检查解释效果。
`NUM_JITTERS` 只影响 128 维描述子，主模型重训时也需重新提取描述子。

> 关于 `DETECT_UPSAMPLE`：调大它能检出更小的人脸，但也可能改变检测框及关键点位置，
> 因而影响几何解释特征，不应只当作性能开关。

**`未找到模型文件` / `模型文件缺少必需字段`**
前者是 `.dat` 或 `.joblib` 缺失，按 §1.2 下载或重训。后者说明 joblib 文件损坏或
是旧版本格式，按缺失文件类型分别重跑 `python src/train_final.py`（主模型）或
`python src/train_geometry.py`（辅助模型）。

**拒识太多、几乎全是「不确定」**
说明输入照片的拍摄条件与训练数据差异较大。可以临时用 `-t 0.4` 看看分布，
但请记住：降低阈值只是让模型「更敢说」，不代表判断更准。

**结果怎么和论文对不上**
本项目复现的论文（喻为栋等，2024）报告融合特征 AUC 0.94~0.98，本项目实测
0.88~0.94，且**融合特征相比纯深度特征没有提升**（论文的核心结论）。差异原因与
完整分析见 `docs/step.md`。主要怀疑是数据不同、折划分协议不同（论文未说明），
以及样本量过小。

**Windows 控制台报 `UnicodeEncodeError: 'gbk' codec`**
GBK 控制台打印不了部分字符。命令前加 `PYTHONIOENCODING=utf-8`。

---

## 相关文件

| 路径 | 作用 |
|---|---|
| `src/face_embedding.py` | 共享底层：图片 → 原图坐标系 68 点及 128 维描述子 |
| `src/extract_features_61.py` | 从 68 个关键点生成语义几何特征 |
| `src/train_final.py` | 训练并保存 128 维深度主模型及拒识阈值表 |
| `src/train_geometry.py` | 训练并保存 61 维几何解释辅助模型 |
| `src/explain.py` | 对几何辅助模型的类别概率进行 SHAP 解释 |
| `src/predict.py` | 推理入口（命令行 + Python 接口 + 拒识机制 + 可选解释） |
| `src/reproduce_paper.py` | 复现论文 4 模型 × 5 特征方案的对比实验 |
| `src/extract_embeddings_128.py` | 批量提取 128 维特征 |
| `src/extract_features_61.py` | 提取 61 维几何特征 |
| `docs/step.md` | 完整复现步骤与实验结论 |
| `docs/inform.md` | 论文要点整理 |
