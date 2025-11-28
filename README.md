# 🚀 my_yolo_platform - All-in-One Visual AI Platform

**my_yolo_platform** 是一个基于 Flask + Ultralytics YOLOv11 构建的轻量级、全流程可视化 AI 训练与推理平台。

它整合了从 **数据标注** -> **模型训练** -> **性能监控** -> **推理检测** 的完整闭环，专为深度学习开发者和科研人员设计，无需编写繁琐代码即可轻松管理 YOLO 任务。

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.9%2B-green)
![YOLO](https://img.shields.io/badge/YOLO-v11-orange)

## ✨ 核心功能 (Features)

### 1. 📊 总览仪表盘 (Dashboard)
- **硬件监控**：实时显示 NVIDIA GPU 占用率、显存、温度以及 CPU/内存状态。
- **全局统计**：统计模型数量、数据集数量、最佳 mAP 记录及磁盘占用。
- **动态图表**：资源使用率的历史波形图。

### 2. 🏷️ 在线数据标注 (Labeling)
- **Web 标注器**：内置 Canvas 标注工具，无需安装 LabelImg。
- **自动保存**：实时保存为 YOLO 格式 TXT 标注文件。
- **一键导出**：自动划分训练集/验证集（8:2），生成 `data.yaml` 并打包为 Zip 下载。

### 3. 🧠 可视化模型训练 (Training)
- **多版本支持**：支持 YOLOv11 n/s/m/l/x 全系列模型。
- **高级配置**：支持自定义 Epochs、Batch、ImgSz，以及 Mosaic、旋转、翻转等数据增强参数。
- **实时监控**：
    - 实时日志流（WebSocket/Polling）。
    - 实时 Loss & mAP 折线图（Chart.js）。
    - 训练进度条与剩余时间估算。
    - 实时预览验证集预测图 (`val_batch0_pred.jpg`)。
- **断点续训**：支持 Resume 功能，从中断处继续训练。
- **后台任务**：全异步多线程处理，页面刷新不中断训练。

### 4. 👁️ 推理与演示 (Inference)
- **模型管理**：自动扫描并加载所有预训练模型及用户自训练模型 (`best.pt`)。
- **多媒体支持**：支持图片和视频上传检测。
- **交互控制**：支持动态调整置信度阈值 (Confidence Slider)。
- **结果导出**：支持一键下载检测后的图片或视频。

### 5. ⚙️ 系统设置 (Settings)
- **缓存清理**：一键清理临时上传文件，释放磁盘空间。
- **任务管理**：查看历史训练记录，一键删除不满意的模型与日志。

---

## 🛠️ 安装与运行 (Installation)

### 环境要求
- Python 3.9+
- CUDA Toolkit (推荐，用于 GPU 加速)
- FFmpeg (用于视频转码)

### 1. 克隆项目
```bash
git clone https://github.com/yourusername/my_yolo_platform.git
cd my_yolo_platform
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```
FFmpeg 安装指南:
```bash
Ubuntu: sudo apt install ffmpeg
Windows: 下载 FFmpeg 解压并将 bin 目录添加到系统环境变量 Path 中。
```
### 3. 启动项目
```bash
python app.py
```
启动成功后，访问浏览器：http://localhost:7860

## 🛠️ 目录结构 (Directory Structure)
```bash
my_yolo_platform/
├── app.py                  # 程序入口
├── config.py               # 全局配置
├── requirements.txt        # 依赖列表
├── services/               # [业务逻辑层]
│   ├── dashboard_service.py # 统计与文件管理
│   ├── system_service.py    # 硬件监控
│   ├── training_service.py  # 训练线程与COCO转换
│   ├── inference_service.py # 推理逻辑
│   └── labeling_service.py  # 标注逻辑
├── routes/                 # [路由控制层]
│   ├── dashboard_routes.py
│   ├── training_routes.py
│   ├── inference_routes.py
│   └── labeling_routes.py
├── templates/              # [前端模板]
│   ├── base.html           # 母版页 (含侧边栏)
│   ├── dashboard.html      # 总览
│   ├── train.html          # 训练配置与监控
│   ├── inference.html      # 推理演示
│   ├── labeling.html       # 标注工具
│   └── settings.html       # 设置
├── static/                 # 静态文件
│   ├── uploads/            # 临时上传区
│   └── results/            # 推理结果区
├── datasets/               # 数据集存放区
└── runs/                   # 训练结果保存区 (YOLO自动生成)
```
## 📖 使用指南 (Quick Start)
```bash
1.准备数据：
进入 "数据标注" 页面，上传原始图片。
在网页上进行画框标注，保存。
点击 "导出训练包"，获得 zip 文件。
2.开始训练：
进入 "模型训练" 页面。
上传刚才下载的 zip 包。
输入任务名称（如 my_v1），选择模型大小（推荐 yolo11m.pt）。
点击 "开始训练"，观察图表和日志。
3.模型推理：
训练完成后，进入 "推理检测" 页面。
在模型下拉框中选择 my_v1 (Trained)。
上传测试图片或视频，查看效果并下载。
```
