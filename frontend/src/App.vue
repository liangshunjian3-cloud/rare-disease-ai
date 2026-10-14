<template>
  <div class="app">
    <!-- 顶部导航 -->
    <header class="header">
      <div class="brand">
        <div class="brand-icon">AI</div>
        <div>
          <h1>Rare Disease AI</h1>
          <p>罕见遗传综合征辅助识别系统</p>
        </div>
      </div>

      <span class="status-tag"><span class="status-dot"></span>研究预览版</span>
    </header>

    <main class="main">
      <!-- 页面介绍 -->
      <section class="hero">
        <div class="eyebrow"><span></span> AI-POWERED ANALYSIS</div>
        <h2>让面部特征分析，<br class="hero-break" />更清晰直观。</h2>
        <p>
          上传清晰的正面照片，获取模型类别概率与前五项几何特征贡献。
        </p>
      </section>

      <!-- 左侧上传，右侧结果 -->
      <section class="workspace">
        <div class="panel upload-panel">
          <div class="panel-heading">
            <div>
              <h3>01 · 图像上传</h3>
              <p>请选择一张清晰的单人面部照片</p>
            </div>
          </div>

          <input
            ref="fileInput"
            type="file"
            accept="image/jpeg,image/png,image/webp"
            class="hidden-input"
            @change="handleFileChange"
          />

          <div
            class="drop-zone"
            :class="{ 'has-image': previewUrl }"
            role="button"
            tabindex="0"
            aria-label="选择一张面部图像"
            @click="fileInput?.click()"
            @keydown.enter.prevent="fileInput?.click()"
            @keydown.space.prevent="fileInput?.click()"
          >
            <img
              v-if="previewUrl"
              :src="previewUrl"
              alt="待分析的面部照片"
              class="preview-image"
            />

            <template v-else>
              <div class="upload-icon">↑</div>
              <h4>选择面部图像</h4>
              <p>点击此处，从电脑中选择图片</p>
              <span class="file-tip">JPG / PNG / WEBP · 最大 10 MB</span>
            </template>
          </div>

          <div v-if="selectedFile" class="file-info">
            <div class="file-name">
              <span class="file-icon">IMG</span>
              <div>
                <strong>{{ selectedFile.name }}</strong>
                <p>{{ formatFileSize(selectedFile.size) }}</p>
              </div>
            </div>

            <button class="text-button" @click="clearImage">
              移除
            </button>
          </div>

          <p v-if="errorMessage" class="error-message">
            {{ errorMessage }}
          </p>

          <button
            class="primary-button"
            :disabled="!selectedFile || loading"
            @click="handleAnalyze"
          >
            <span v-if="loading" class="spinner"></span>
            {{ loading ? '正在分析...' : '开始分析 →' }}
          </button>

          <p class="helper-text">
            上传图像将由 128 维深度分类器预测，并由几何辅助模型提供 SHAP 特征解释。
          </p>
        </div>

        <!-- 右侧结果 -->
        <div class="panel result-panel">
          <div class="panel-heading">
            <div>
              <h3>02 · 分析结果</h3>
              <p>查看类别预测与模型解释</p>
            </div>
            <span class="result-tag">
              {{ result ? '分析完成' : '等待分析' }}
            </span>
          </div>

          <div v-if="!result && !loading" class="empty-result">
            <div class="empty-icon">◎</div>
            <h4>等待图像分析</h4>
            <p>上传图片并点击“开始分析”，这里将展示预测结果。</p>
          </div>

          <div v-else-if="loading" class="empty-result">
            <div class="spinner large-spinner"></div>
            <h4>正在处理图像</h4>
            <p>正在运行人脸检测、模型预测和几何特征解释……</p>
          </div>

          <template v-else-if="result">
            <div class="prediction-card">
              <div
                class="prediction-label"
                :class="{ 'is-rejected': result.prediction.rejected }"
              >
                {{ result.prediction.rejected ? '低于拒识阈值 · 建议人工复核' : '深度模型预测' }}
              </div>
              <h2>{{ result.prediction.label }}</h2>
              <p>
                最高类别概率
                <strong>
                  {{ (result.prediction.probability * 100).toFixed(1) }}%
                </strong>
              </p>
              <p v-if="result.prediction.threshold">
                拒识阈值：{{ (result.prediction.threshold * 100).toFixed(0) }}%
              </p>
            </div>

            <div class="section-title">
              <h4>候选类别排序</h4>
              <span>Top 3</span>
            </div>

            <div class="top-list">
              <div
                v-for="(item, index) in result.topK"
                :key="item.label"
                class="top-item"
              >
                <div class="top-item-heading">
                  <span class="rank">{{ index + 1 }}</span>
                  <span class="class-name">{{ item.label }}</span>
                  <strong>
                    {{ (item.probability * 100).toFixed(1) }}%
                  </strong>
                </div>
                <div class="progress-track">
                  <div
                    class="progress-bar"
                    :style="{ width: `${item.probability * 100}%` }"
                  ></div>
                </div>
              </div>
            </div>

            <div class="section-title explanation-title">
              <h4>几何辅助模型 SHAP 特征贡献</h4>
              <span>{{ result.explanation ? 'TOP 5' : '未请求解释' }}</span>
            </div>

            <p class="explanation-intro">
              <template v-if="result.explanation">
                针对深度模型预测类别「{{ result.explanation.classCn }}」的几何辅助解释。
                几何模型自身预测：{{ result.explanation.geometryModelPredictionCn }}
                （{{ (result.explanation.geometryModelPredictionProbability * 100).toFixed(1) }}%）。
              </template>
              <template v-else>
                本次请求未启用 SHAP 几何特征解释。
              </template>
            </p>

            <div class="feature-list">
              <div
                v-for="item in result.explanations"
                :key="item.name"
                class="feature-item"
              >
                <div class="feature-heading">
                  <span>{{ item.name }}</span>
                  <strong
                    :class="item.shapValue >= 0 ? 'positive' : 'negative'"
                  >
                    {{ item.shapValue > 0 ? '+' : '' }}{{
                      item.shapValue.toFixed(3)
                    }}
                  </strong>
                </div>

                <div class="contribution-track">
                  <div
                    class="contribution-bar"
                    :class="
                      item.shapValue >= 0 ? 'positive-bar' : 'negative-bar'
                    "
                    :style="{
                      width: `${Math.abs(item.shapValue) / maxAbs * 100}%`
                    }"
                  ></div>
                </div>

                <p>当前特征值：{{ item.value }}</p>
              </div>
            </div>
            <p class="explanation-intro">
              {{ result.warning }}
            </p>
          </template>
        </div>
      </section>

      <!-- 底部提示 -->
      <footer class="disclaimer">
        <strong>重要提示</strong>
        <p>
          本系统用于课程设计和计算机辅助识别研究。模型结果不构成医学诊断，
          不能替代专业医生的临床评估。几何特征解释不是深度主模型的直接归因。
        </p>
      </footer>
    </main>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, ref } from 'vue'
import { predictFace } from './api/prediction'

const fileInput = ref(null)
const selectedFile = ref(null)
const previewUrl = ref('')
const loading = ref(false)
const errorMessage = ref('')
const result = ref(null)



const maxAbs = computed(() => {
  const values = (result.value?.explanations ?? [])
    .map(item => Math.abs(item.shapValue))

  return Math.max(...values, 0.001)
})

function formatFileSize(bytes) {
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`
  }

  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
}

function handleFileChange(event) {
  const file = event.target.files?.[0]

  if (!file) return

  const allowedTypes = [
    'image/jpeg',
    'image/png',
    'image/webp'
  ]

  if (!allowedTypes.includes(file.type)) {
    clearImage()
    errorMessage.value = '文件格式不支持，请选择 JPG、PNG 或 WEBP 图片。'
    return
  }

  if (file.size > 10 * 1024 * 1024) {
    clearImage()
    errorMessage.value = '图片大小不能超过 10 MB。'
    return
  }

  releasePreview()

  selectedFile.value = file
  previewUrl.value = URL.createObjectURL(file)
  errorMessage.value = ''
  result.value = null
}

function releasePreview() {
  if (previewUrl.value) {
    URL.revokeObjectURL(previewUrl.value)
    previewUrl.value = ''
  }
}

function clearImage() {
  releasePreview()

  selectedFile.value = null
  result.value = null
  errorMessage.value = ''

  if (fileInput.value) {
    fileInput.value.value = ''
  }
}

async function handleAnalyze() {
  if (!selectedFile.value || loading.value) return

  loading.value = true
  errorMessage.value = ''
  result.value = null

  try {
    // 将图片发送给 FastAPI，并接收预测结果
    result.value = await predictFace(selectedFile.value)

  } catch (error) {
    console.error('预测请求失败：', error)

    errorMessage.value =
      error.response?.data?.detail ||
      '分析失败，请检查后端服务是否启动。'

  } finally {
    loading.value = false
  }
}

onBeforeUnmount(() => {
  releasePreview()
})
</script>

<style>
:root {
  font-family: Inter, "Microsoft YaHei", "PingFang SC", sans-serif;
  color: #182230;
  background: #f4f7fb;
  font-synthesis: none;
  text-rendering: optimizeLegibility;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  min-width: 320px;
}

button,
input {
  font: inherit;
}

button {
  cursor: pointer;
}

.app {
  min-height: 100vh;
}

.header {
  height: 84px;
  background: #ffffff;
  border-bottom: 1px solid #e9edf3;
  padding: 0 7%;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.brand {
  display: flex;
  align-items: center;
  gap: 13px;
}

.brand-icon {
  width: 43px;
  height: 43px;
  border-radius: 13px;
  display: grid;
  place-items: center;
  color: #ffffff;
  background: #245fe7;
  font-weight: 800;
  letter-spacing: -1px;
}

.brand h1 {
  font-size: 17px;
  margin: 0 0 4px;
}

.brand p {
  font-size: 12px;
  margin: 0;
  color: #7c8797;
}

.status-tag,
.result-tag {
  border-radius: 20px;
  background: #edf3ff;
  color: #285ac7;
  font-size: 12px;
  padding: 7px 12px;
}

.main {
  max-width: 1250px;
  width: 88%;
  margin: 0 auto;
  padding: 44px 0 32px;
}

.hero {
  margin-bottom: 30px;
}

.eyebrow {
  color: #315fce;
  letter-spacing: 2px;
  font-weight: 700;
  font-size: 11px;
  margin-bottom: 12px;
}

.hero h2 {
  font-size: clamp(27px, 3vw, 36px);
  letter-spacing: -1px;
  margin: 0 0 12px;
}

.hero p {
  font-size: 14px;
  color: #687589;
  margin: 0;
  line-height: 1.8;
}

.workspace {
  display: grid;
  grid-template-columns: minmax(0, 0.92fr) minmax(0, 1.08fr);
  gap: 22px;
  align-items: stretch;
}

.panel {
  background: white;
  border: 1px solid #e7ebf1;
  border-radius: 18px;
  padding: 25px;
  box-shadow: 0 8px 30px rgb(30 55 90 / 4%);
  min-width: 0;
}

.panel-heading {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  margin-bottom: 23px;
}

.panel-heading h3 {
  margin: 0 0 7px;
  font-size: 16px;
}

.panel-heading p {
  color: #8993a3;
  font-size: 12px;
  margin: 0;
}

.hidden-input {
  display: none;
}

.drop-zone {
  height: 310px;
  border: 1px dashed #bdcce4;
  border-radius: 13px;
  background: #f8faff;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 18px;
  cursor: pointer;
  overflow: hidden;
  transition: border-color 0.2s, background 0.2s;
}

.drop-zone:hover {
  border-color: #386be3;
  background: #f3f7ff;
}

.drop-zone.has-image {
  border-style: solid;
  border-color: #dfe7f2;
  background: #f6f8fb;
}

.upload-icon {
  width: 52px;
  height: 52px;
  display: grid;
  place-items: center;
  background: #e7efff;
  color: #2c62d9;
  border-radius: 15px;
  font-size: 29px;
  margin-bottom: 18px;
}

.drop-zone h4 {
  font-size: 15px;
  margin: 0 0 9px;
}

.drop-zone p {
  color: #7e8a9c;
  font-size: 12px;
  margin: 0 0 18px;
  text-align: center;
}

.file-tip {
  color: #9aa3b0;
  font-size: 11px;
}

.preview-image {
  width: 100%;
  height: 100%;
  object-fit: contain;
}

.file-info {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 13px 0 4px;
}

.file-name {
  min-width: 0;
  display: flex;
  align-items: center;
  gap: 10px;
}

.file-icon {
  flex-shrink: 0;
  color: #315fce;
  background: #eaf0ff;
  font-size: 10px;
  font-weight: 700;
  border-radius: 8px;
  padding: 10px 7px;
}

.file-name strong {
  font-size: 12px;
  overflow-wrap: anywhere;
}

.file-name p {
  margin: 4px 0 0;
  color: #8c97a7;
  font-size: 11px;
}

.text-button {
  border: 0;
  color: #cf4750;
  background: transparent;
  font-size: 12px;
}

.primary-button {
  margin-top: 20px;
  height: 47px;
  border: 0;
  border-radius: 10px;
  width: 100%;
  background: #285fe0;
  color: white;
  font-weight: 600;
  font-size: 13px;
  transition: background 0.2s;
}

.primary-button:hover:not(:disabled) {
  background: #174bc5;
}

.primary-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.helper-text {
  color: #98a1af;
  text-align: center;
  font-size: 11px;
  line-height: 1.7;
  margin: 13px 0 0;
}

.error-message {
  color: #c43c44;
  font-size: 12px;
  line-height: 1.6;
}

.empty-result {
  min-height: 400px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  text-align: center;
  padding: 30px;
}

.empty-icon {
  color: #9aafd9;
  font-size: 57px;
  margin-bottom: 14px;
}

.empty-result h4 {
  margin: 0 0 10px;
  font-size: 15px;
}

.empty-result p {
  color: #8993a3;
  font-size: 12px;
  line-height: 1.8;
  max-width: 290px;
  margin: 0;
}

.spinner {
  width: 17px;
  height: 17px;
  display: inline-block;
  border: 2px solid rgb(255 255 255 / 45%);
  border-top-color: white;
  border-radius: 50%;
  animation: spin 0.7s linear infinite;
  vertical-align: middle;
  margin-right: 7px;
}

.large-spinner {
  width: 34px;
  height: 34px;
  margin: 0 0 22px;
  border-color: #d8e3fb;
  border-top-color: #285fe0;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

.prediction-card {
  padding: 21px;
  background: linear-gradient(125deg, #edf3ff, #f5f8ff);
  border: 1px solid #e0eaff;
  border-radius: 13px;
}

.prediction-label {
  color: #5370a5;
  font-size: 11px;
  margin-bottom: 11px;
}

.prediction-card h2 {
  margin: 0 0 12px;
  font-size: 25px;
  color: #1f48a3;
}

.prediction-card p {
  margin: 0;
  font-size: 12px;
  color: #63738f;
}

.prediction-card strong {
  font-size: 18px;
  color: #245bcf;
  margin-left: 5px;
}

.section-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin: 26px 0 17px;
}

.section-title h4 {
  margin: 0;
  font-size: 13px;
}

.section-title > span {
  font-size: 11px;
  color: #8290a5;
}

.top-list {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.top-item-heading,
.feature-heading {
  display: flex;
  align-items: center;
  gap: 9px;
}

.top-item-heading {
  margin-bottom: 8px;
}

.rank {
  width: 21px;
  height: 21px;
  display: grid;
  place-items: center;
  border-radius: 6px;
  background: #f0f3f8;
  color: #657287;
  font-size: 10px;
}

.class-name,
.feature-heading > span {
  flex: 1;
  font-size: 12px;
}

.top-item-heading strong {
  font-size: 12px;
}

.progress-track,
.contribution-track {
  width: 100%;
  background: #edf0f5;
  border-radius: 20px;
  overflow: hidden;
}

.progress-track {
  height: 6px;
}

.progress-bar {
  height: 100%;
  background: #3d70e8;
  border-radius: 20px;
  transition: width 0.4s;
}

.explanation-title {
  margin-bottom: 9px;
}

.explanation-intro {
  margin: 0 0 17px;
  font-size: 11px;
  line-height: 1.8;
  color: #8792a3;
}

.feature-list {
  display: flex;
  flex-direction: column;
  gap: 17px;
}

.feature-heading strong {
  font-size: 12px;
  font-variant-numeric: tabular-nums;
}

.positive {
  color: #21866b;
}

.negative {
  color: #d05c62;
}

.contribution-track {
  height: 5px;
  margin: 9px 0 5px;
}

.contribution-bar {
  height: 100%;
  border-radius: 20px;
}

.positive-bar {
  background: #34a783;
}

.negative-bar {
  background: #e17b80;
}

.feature-item > p {
  font-size: 10px;
  color: #98a1af;
  margin: 0;
}

.disclaimer {
  margin-top: 24px;
  padding: 17px 20px;
  background: #fff;
  border: 1px solid #e7ebf1;
  border-radius: 12px;
}

.disclaimer strong {
  font-size: 12px;
  color: #69768b;
}

.disclaimer p {
  font-size: 11px;
  line-height: 1.8;
  color: #8a95a4;
  margin: 7px 0 0;
}

.header {
  position: relative;
  z-index: 1;
  height: 78px;
  padding-inline: max(6%, calc((100% - 1320px) / 2));
  background: rgb(255 255 255 / 88%);
  border-color: #e8edf5;
  box-shadow: 0 4px 24px rgb(33 55 91 / 3%);
  backdrop-filter: blur(16px);
}

.brand {
  gap: 12px;
}

.brand-icon {
  width: 44px;
  height: 44px;
  border-radius: 14px;
  background: linear-gradient(145deg, #347af4, #2153cc);
  box-shadow: 0 7px 16px rgb(43 99 218 / 24%);
}

.brand h1 {
  color: #16243b;
  font-size: 16px;
  font-weight: 700;
  letter-spacing: -0.35px;
}

.brand p {
  color: #8390a3;
}

.status-tag {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  color: #526581;
  background: #f2f5fa;
  border: 1px solid #e8edf5;
  font-size: 11px;
}

.status-dot,
.eyebrow > span {
  width: 7px;
  height: 7px;
  display: inline-block;
  border-radius: 50%;
  background: #42ad83;
  box-shadow: 0 0 0 3px rgb(66 173 131 / 13%);
}

.main {
  max-width: 1240px;
  padding-top: 54px;
}

.hero {
  position: relative;
  margin: 0 0 34px;
  padding: 34px 38px;
  overflow: hidden;
  border: 1px solid #e4ebf6;
  border-radius: 22px;
  background:
    radial-gradient(ellipse at 86% 10%, rgb(98 147 248 / 18%), transparent 36%),
    linear-gradient(112deg, #fff 5%, #f7faff 100%);
  box-shadow: 0 12px 34px rgb(39 66 109 / 5%);
}

.hero::after {
  position: absolute;
  right: 9%;
  bottom: -108px;
  width: 260px;
  height: 260px;
  border: 1px solid rgb(71 119 213 / 10%);
  border-radius: 50%;
  content: "";
  box-shadow: 0 0 0 28px rgb(71 119 213 / 4%), 0 0 0 58px rgb(71 119 213 / 3%);
  pointer-events: none;
}

.eyebrow {
  display: flex;
  align-items: center;
  gap: 9px;
  margin-bottom: 17px;
  color: #426ab5;
  font-size: 10px;
  letter-spacing: 1.8px;
}

.eyebrow > span {
  background: #4e82e5;
  box-shadow: 0 0 0 3px rgb(78 130 229 / 13%);
}

.hero h2 {
  position: relative;
  z-index: 1;
  margin-bottom: 12px;
  color: #182b49;
  font-size: clamp(29px, 3.5vw, 40px);
  font-weight: 750;
  letter-spacing: -1.5px;
  line-height: 1.3;
}

.hero p {
  position: relative;
  z-index: 1;
  color: #73829a;
  font-size: 13px;
}

.workspace {
  gap: 20px;
}

.panel {
  padding: 26px;
  border-color: #e7ecf4;
  border-radius: 20px;
  box-shadow: 0 12px 36px rgb(33 55 91 / 5%);
  transition: box-shadow 0.25s, transform 0.25s;
}

.panel:hover {
  box-shadow: 0 16px 42px rgb(33 55 91 / 8%);
  transform: translateY(-2px);
}

.panel-heading {
  margin-bottom: 21px;
}

.panel-heading h3 {
  color: #22324b;
  font-size: 15px;
  font-weight: 700;
}

.panel-heading p {
  color: #8996a9;
}

.drop-zone {
  position: relative;
  height: 306px;
  border: 1px dashed #b9c9e2;
  border-radius: 16px;
  background:
    radial-gradient(circle at 50% 40%, rgb(228 238 255 / 65%), transparent 44%),
    #f9fbff;
  outline: none;
  transition: border-color 0.2s, background 0.2s, box-shadow 0.2s;
}

.drop-zone:hover,
.drop-zone:focus-visible {
  border-color: #5483e4;
  background-color: #f4f8ff;
  box-shadow: inset 0 0 0 3px rgb(84 131 228 / 7%);
}

.drop-zone.has-image {
  background: #f2f5fa;
}

.upload-icon {
  width: 58px;
  height: 58px;
  color: #fff;
  background: linear-gradient(145deg, #5e91f2, #3269d9);
  border-radius: 18px;
  box-shadow: 0 10px 22px rgb(59 111 220 / 23%);
  font-size: 27px;
}

.drop-zone h4 {
  color: #263750;
  font-size: 14px;
}

.drop-zone p {
  color: #8492a7;
}

.file-info {
  margin-top: 13px;
  padding: 11px 12px;
  border: 1px solid #e9eef5;
  border-radius: 12px;
  background: #fbfcfe;
}

.text-button {
  padding: 7px 9px;
  border-radius: 8px;
  transition: background 0.2s;
}

.text-button:hover {
  background: #fff0f0;
}

.primary-button {
  height: 49px;
  margin-top: 18px;
  border-radius: 12px;
  background: linear-gradient(110deg, #3678ed, #285bd4);
  box-shadow: 0 8px 18px rgb(46 100 216 / 19%);
  font-size: 13px;
  transition: box-shadow 0.2s, transform 0.2s, background 0.2s;
}

.primary-button:hover:not(:disabled) {
  background: linear-gradient(110deg, #2869df, #204fc4);
  box-shadow: 0 11px 22px rgb(46 100 216 / 26%);
  transform: translateY(-1px);
}

.primary-button:focus-visible,
.text-button:focus-visible {
  outline: 3px solid rgb(62 119 231 / 32%);
  outline-offset: 3px;
}

.primary-button:disabled {
  box-shadow: none;
}

.result-tag {
  white-space: nowrap;
  font-size: 10px;
  font-weight: 650;
}

.prediction-card {
  position: relative;
  padding: 23px 24px;
  overflow: hidden;
  border-color: #dce8fc;
  border-radius: 16px;
  background:
    radial-gradient(circle at 92% 0%, rgb(98 147 248 / 17%), transparent 34%),
    linear-gradient(125deg, #f1f6ff, #f8faff);
}

.prediction-card::after {
  position: absolute;
  top: -35px;
  right: -20px;
  width: 128px;
  height: 128px;
  border: 1px solid rgb(61 110 210 / 12%);
  border-radius: 50%;
  content: "";
  box-shadow: 0 0 0 18px rgb(61 110 210 / 4%);
  pointer-events: none;
}

.prediction-label {
  display: inline-flex;
  padding: 6px 9px;
  color: #4169af;
  background: #e3edff;
  border-radius: 7px;
  font-weight: 650;
}

.prediction-label.is-rejected {
  color: #966613;
  background: #fff2d9;
}

.prediction-card h2 {
  position: relative;
  z-index: 1;
  margin-bottom: 9px;
  color: #1e468d;
  font-size: 25px;
  font-weight: 750;
}

.prediction-card p + p {
  margin-top: 7px;
  color: #8794a8;
  font-size: 11px;
}

.prediction-card strong {
  color: #285ec9;
  font-size: 20px;
  font-variant-numeric: tabular-nums;
}

.section-title {
  margin-top: 25px;
}

.section-title h4 {
  color: #293850;
  font-size: 13px;
  font-weight: 700;
}

.section-title > span {
  padding: 5px 8px;
  border: 1px solid #e8edf5;
  border-radius: 7px;
  background: #f8faff;
  color: #77859b;
  font-size: 9px;
  font-weight: 650;
  letter-spacing: 0.4px;
}

.top-list {
  gap: 14px;
}

.rank {
  width: 23px;
  height: 23px;
  border: 1px solid #e6ebf3;
  background: #f6f8fb;
  font-weight: 700;
}

.top-item:first-child .rank {
  color: #fff;
  background: #3975e5;
  border-color: #3975e5;
}

.class-name,
.feature-heading > span {
  color: #43516a;
  font-size: 12px;
  font-weight: 550;
}

.top-item-heading strong {
  color: #354a69;
  font-variant-numeric: tabular-nums;
}

.progress-track {
  height: 7px;
  background: #edf1f7;
}

.progress-bar {
  background: linear-gradient(90deg, #72a0f5, #3f73df);
}

.explanation-title {
  margin-top: 29px;
}

.explanation-intro {
  color: #78869a;
  font-size: 11px;
}

.feature-list {
  gap: 10px;
}

.feature-item {
  padding: 12px 13px 10px;
  border: 1px solid #edf0f5;
  border-radius: 12px;
  background: #fcfdff;
  transition: border-color 0.2s, background 0.2s, transform 0.2s;
}

.feature-item:hover {
  border-color: #dce6f7;
  background: #f9fbff;
  transform: translateX(2px);
}

.feature-heading strong {
  font-size: 12px;
  font-variant-numeric: tabular-nums;
}

.contribution-track {
  height: 6px;
  margin: 9px 0 6px;
  background: #edf1f6;
}

.positive-bar {
  background: linear-gradient(90deg, #73c7a9, #36a77f);
}

.negative-bar {
  background: linear-gradient(90deg, #f0a2a2, #d86c73);
}

.feature-item > p {
  color: #909bad;
}

.disclaimer {
  margin-top: 22px;
  padding: 16px 19px;
  border-color: #e7ebf2;
  border-radius: 14px;
  background: rgb(255 255 255 / 78%);
}

.disclaimer strong {
  color: #64738a;
}

.error-message {
  padding: 10px 12px;
  border: 1px solid #f3d6d8;
  border-radius: 10px;
  background: #fff8f8;
}

@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    scroll-behavior: auto !important;
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}

@media (max-width: 800px) {
  .main {
    padding-top: 34px;
  }

  .hero {
    padding: 29px 28px;
  }

  .header {
    padding: 0 5%;
  }

  .main {
    width: 92%;
    padding-top: 30px;
  }

  .workspace {
    grid-template-columns: 1fr;
  }

  .empty-result {
    min-height: 230px;
  }

  .drop-zone {
    height: 280px;
  }
}

@media (max-width: 440px) {
  .header {
    height: 75px;
  }

  .brand h1 {
    font-size: 14px;
  }

  .brand p {
    font-size: 10px;
  }

  .status-tag {
    font-size: 10px;
    padding: 6px 8px;
  }

  .panel {
    padding: 18px;
  }

  .hero h2 {
    font-size: 27px;
  }

  .hero {
    padding: 25px 21px;
    border-radius: 18px;
  }

  .hero-break {
    display: none;
  }

  .status-tag {
    max-width: 104px;
    white-space: nowrap;
  }

  .panel {
    border-radius: 16px;
  }
}
</style>