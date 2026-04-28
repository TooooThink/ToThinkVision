# ToThinkVision — 四合一通用视觉结构化引擎

输入任意图片（UI截图/游戏场景/动漫/实拍/AI图）或视频，自动输出结构化数据：物体轮廓、坐标、层级、颜色、文字、运动轨迹、空间关系、时序信息。

再从 4 大模块中选择一种导出格式，一键生成对应结果。

---

## 一、快速上手

### 最快体验（无需 GPU / 模型）

```bash
# 1. 安装依赖（核心包约 200MB）
pip install pydantic pydantic-settings numpy scipy pytest pillow opencv-python-headless fastapi uvicorn python-multipart

# 2. 启动服务（mock 模式，用模拟数据跑通全流程）
MOCK_MODE=true uvicorn app.main:app --host 0.0.0.0 --port 8000

# 3. 浏览器打开 http://localhost:8000 即可使用
```

### 完整部署（使用真实 AI 模型）

```bash
# 1. 安装全部依赖
pip install -r requirements.txt

# 2. 运行安装脚本（自动下载模型权重）
chmod +x setup.sh && ./setup.sh

# 3. 启动服务
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 4. 浏览器访问 http://localhost:8000
```

> **注意：** 首次启动时若未找到模型权重文件，系统会自动降级到 mock 模式，不会报错。

---

## 二、功能一览

```
上传文件 → 自动识别图片/视频 → 执行 AI 分析 → 统一中间层 JSON → 导出目标格式
```

### 四大导出模块

| 模块 | 输入 | 输出 | 适用场景 |
|------|------|------|----------|
| **UI 解构** | APP/网页/AI UI 截图 | Figma JSON / HTML+CSS / UI JSON | 设计还原、组件提取 |
| **游戏场景** | 游戏截图/原画/视频 | Unity JSON / UE JSON / 碰撞盒 | 场景搭建、碰撞体生成 |
| **视频动效** | 游戏/演示/动效视频 | AE 关键帧 / 轨迹 CSV / PR 标记 | 动效分析、剪辑辅助 |
| **具身智能** | VR/机器人操作视频 | 机器人动作序列 / 3D 位姿 CSV | 机器人训练数据生成 |

### 13 种导出格式

- `figma_json` — Figma 文档结构
- `html_css` — 带内联 CSS 的自包含 HTML
- `ui_json` — 简化 UI 组件 JSON
- `unity_json` — Unity GameObject + Collider + Rigidbody
- `ue_json` — Unreal Engine Actor 结构
- `collision_json` — 纯碰撞盒数据
- `ae_keyframes` — After Effects 关键帧时间线
- `video_trajectory` — 逐帧物体轨迹 CSV
- `pr_markers` — Premiere Pro 章节标记
- `embodied_json` — 场景 + 交互序列
- `robot_action` — 机器人接近/抓取动作序列
- `pose_csv` — 逐帧 3D 位姿 CSV
- `full_json` — 完整中间层结构化 JSON

---

## 三、代码结构详解

```
ToThinkVision/
├── app/                              # 后端核心代码
│   ├── __init__.py
│   │
│   ├── main.py                       # ★ 入口文件
│   │   └── FastAPI 服务 + REST API
│   │       ├── GET  /               → 返回前端页面
│   │       ├── GET  /api/formats    → 列出所有导出格式
│   │       ├── POST /api/process    → 上传 + 处理 + 导出
│   │       ├── GET  /api/download   → 下载输出文件
│   │       └── GET  /api/health     → 健康检查
│   │
│   ├── config.py                     # 全局配置
│   │   └── Settings 类（环境变量 → 路径/阈值/设备）
│   │       └── TTV_MOCK_MODE / TTV_DEVICE / TTV_MAX_VIDEO_FRAMES 等
│   │
│   ├── schemas.py                    # ★ 统一中间层 JSON Schema
│   │   └── Pydantic 模型定义
│   │       ├── StructuredObject     → 单个物体：id/bbox/颜色/文字/轨迹/关系/交互
│   │       ├── StructuredOutput     → 完整输出：元数据 + 物体列表
│   │       ├── ObjectType            → 物体类型枚举（UI/Game/Video/Embodied/Generic）
│   │       └── ExportFormat          → 13 种导出格式枚举
│   │
│   ├── preprocessor.py               # 输入层：文件预处理
│   │   ├── is_image() / is_video()  → 类型判断
│   │   ├── preprocess_image()       → 图片加载 + 缩放
│   │   ├── extract_frames()         → 视频拆帧（FFmpeg / OpenCV 回退）
│   │   └── cleanup_frames()         → 清理临时帧文件
│   │
│   ├── pipeline.py                   # ★ 主流程编排
│   │   ├── process_file()           → 入口：判断图片/视频，分发处理
│   │   ├── _process_image()         → 单张图片：分割+检测+OCR+深度+合并+关系
│   │   ├── _process_video()         → 视频：逐帧处理 + 帧间追踪
│   │   ├── _merge_detections()      → 将分割/检测/OCR 结果合并为 StructuredObject
│   │   ├── _compute_relations()     → 计算物体间关系（碰撞/相对位置/层级）
│   │   └── classify_object_type()   → 标签 → ObjectType 分类
│   │
│   ├── models/                       # AI 模型封装层（每个都有 mock 回退）
│   │   ├── segmentor.py             → SAM 分割 → 返回 mask + bbox + 置信度
│   │   ├── detector.py              → Grounding DINO 开放词汇检测
│   │   ├── ocr_engine.py            → PaddleOCR 文字识别
│   │   ├── depth_estimator.py       → Depth Anything 深度估计
│   │   └── tracker.py               → 帧间 IoU 追踪 + 匈牙利匹配
│   │
│   ├── exporters/                    # 导出层
│   │   ├── base.py                  → Exporter 基类（save_json / 路径生成）
│   │   ├── ui_exporter.py           → 模块1: UI → Figma/HTML/UI JSON
│   │   ├── game_exporter.py         → 模块2: 游戏 → Unity/UE/Collision
│   │   ├── video_exporter.py        → 模块3: 视频 → AE/Trajectory/PR Markers
│   │   └── embodied_exporter.py     → 模块4: 具身 → Robot/Pose/Action
│   │
│   └── utils/                        # 工具函数
│       ├── color.py                 → 提取物体主色（量化 + 频次统计）
│       ├── geometry.py              → 碰撞检测、相对方位判断、z-index 计算
│       └── io.py                    → JSON 读写
│
├── static/                           # 前端
│   ├── index.html                   → 单页应用：拖拽上传 + 格式选择 + 进度 + 下载
│   └── style.css                    → 暗色主题样式
│
├── tests/                            # 测试（40 个用例）
│   ├── test_schemas.py              → 数据结构验证（8 个）
│   ├── test_exporters.py            → 12 种导出格式验证（12 个）
│   └── test_pipeline.py             → Tracker + Geometry + Color + 端到端（20 个）
│
├── uploads/                          # 上传文件暂存目录（自动创建）
├── outputs/                          # 导出输出目录
├── requirements.txt                  # Python 依赖清单
├── setup.sh                          # 一键安装脚本
├── README.md                         # 英文文档
└── README_CN.md                      # ← 你正在看的中文文档
```

---

## 四、如何使用

### 方式一：Web 界面（推荐）

1. 启动服务后打开 `http://localhost:8000`
2. 拖拽或点击上传图片/视频文件
3. 选择导出类别和格式
4. 选择检测模式（general/ui/game/video/embodied）
5. 点击「Process & Export」
6. 等待处理完成，点击下载按钮

### 方式二：API 调用

```bash
# 列出所有导出格式
curl http://localhost:8000/api/formats

# 上传图片 + 导出 Figma JSON
curl -X POST http://localhost:8000/api/process \
  -F "file=@screenshot.png" \
  -F "export_format=figma_json" \
  -F "mode=ui"

# 上传视频 + 导出 AE 关键帧
curl -X POST http://localhost:8000/api/process \
  -F "file=@demo.mp4" \
  -F "export_format=ae_keyframes" \
  -F "mode=video"

# 下载结果
curl -O http://localhost:8000/api/download/screenshot_figma.json
```

### 方式三：Python 直接调用

```python
from app.pipeline import process_file
from app.exporters.ui_exporter import UIExporter
from app.schemas import ExportFormat

# 处理图片
result = process_file("screenshot.png", mode="ui")

# 导出 Figma JSON
exporter = UIExporter(ExportFormat.FIGMA_JSON)
output_path = exporter.export(result)
print(f"导出到: {output_path}")

# 查看结果
import json
print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
```

---

## 五、中间层 JSON 数据结构

每个检测到的物体包含以下字段：

```json
{
  "id": "obj_0001",
  "label": "ui_button",
  "label_custom": "提交按钮",
  "confidence": 0.92,

  "bbox": { "x": 100, "y": 200, "w": 120, "h": 40 },
  "contour": [{ "x": 100, "y": 200 }, { "x": 220, "y": 200 }, ...],

  "bbox_3d": { "x": 160, "y": 220, "z": 2.5 },
  "depth_value": 180.0,

  "dominant_color": "#3b82f6",
  "color_palette": ["#3b82f6", "#1d4ed8"],
  "z_index": 5,

  "text_content": "Submit",
  "text_confidence": 0.95,

  "temporal": {
    "frame_index": 10,
    "appear_frame": 0,
    "disappear_frame": -1,
    "trajectory": [{ "x": 160, "y": 220, "t": 0 }, ...],
    "velocity": { "vx": 0.5, "vy": 0.2 }
  },

  "relations": {
    "parent_id": null,
    "collision_with": ["obj_0002"],
    "relative_positions": [
      { "target_id": "obj_0002", "relation": "above" }
    ]
  },

  "interaction": {
    "type": "clickable",
    "clickable": true
  }
}
```

---

## 六、如何进一步开发

### 6.1 接入真实 AI 模型

当前模型封装层已预留接口，只需下载权重文件即可启用：

```python
# app/models/segmentor.py 中的 _get_segmentor() 函数
# 下载 SAM 权重到 MODEL_CACHE_DIR 后自动切换

# app/models/detector.py 中的 _get_detector() 函数
# 下载 Grounding DINO 权重后自动切换

# app/models/ocr_engine.py 中的 _get_ocr() 函数
# 安装 paddlepaddle + paddleocr 后自动切换

# app/models/depth_estimator.py 中的 _get_depth_model() 函数
# transformers 库安装后自动切换
```

### 6.2 添加新模型

在 `app/models/` 下新建文件，实现相同接口，然后在 `pipeline.py` 中替换调用：

```python
# 例：添加 YOLO 检测器
# app/models/yolo_detector.py
def detect_objects(img, mode="general"):
    # 你的 YOLO 实现
    return [{"bbox": [...], "label": "xxx", "confidence": 0.9}]
```

### 6.3 添加新导出格式

1. 在 `app/schemas.py` 的 `ExportFormat` 枚举中添加新格式
2. 在 `app/exporters/` 下新建导出器，继承 `BaseExporter`
3. 在 `app/main.py` 的 `EXPORT_FORMAT_MAP` 中注册
4. 在 `static/index.html` 的 `FORMAT_MAP` 中添加前端选项

```python
# 例：添加 Blender 导出
class BlenderExporter(BaseExporter):
    format_name = "blender"
    file_extension = ".py"

    def export(self, data: StructuredOutput) -> Path:
        # 生成 Blender Python 脚本
        ...
```

### 6.4 优化追踪算法

当前使用 IoU + 匈牙利匹配的简单追踪。可替换为：

- **DeepSORT** — 加入外观特征匹配
- **ByteTrack** — 利用低分检测提升追踪稳定性
- **AOT** — 半监督视频目标分割追踪

只需修改 `app/models/tracker.py` 中的 `ObjectTracker` 类。

### 6.5 添加更多 UI 识别逻辑

在 `pipeline.py` 的 `classify_object_type()` 和 `_compute_relations()` 中扩展：

- 根据布局推断组件层级（父子关系）
- 根据颜色/圆角判断按钮状态
- 根据文字排列推断列表/表格结构

### 6.6 支持批量处理

在 `app/main.py` 添加批量上传端点：

```python
@app.post("/api/batch_process")
async def batch_process(files: list[UploadFile] = File(...), ...):
    results = []
    for f in files:
        result = process_file(...)
        results.append(result)
    return results
```

### 6.7 前端增强

- 添加结果可视化（Canvas/SVG 叠加标注框）
- 支持在线编辑结构化数据
- 添加对比视图（原图 vs 重建效果）

### 6.8 部署上线

```bash
# Docker 部署
# Dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Nginx 反向代理
# server {
#     listen 80;
#     location / {
#         proxy_pass http://127.0.0.1:8000;
#     }
# }
```

---

## 七、环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TTV_MOCK_MODE` | `false` | 设为 `true` 使用模拟数据（无 GPU 也能跑） |
| `TTV_DEVICE` | `cuda` | 模型推理设备：`cuda` / `cpu` / `mps` |
| `TTV_MAX_VIDEO_FRAMES` | `300` | 视频最大拆帧数 |
| `TTV_MODEL_CACHE_DIR` | `~/.cache/tothinkvision` | 模型权重存储路径 |
| `TTV_MAX_UPLOAD_MB` | `500` | 最大上传文件大小（MB） |

---

## 八、运行测试

```bash
# 运行全部测试（mock 模式）
MOCK_MODE=true pytest tests/ -v

# 按模块运行
MOCK_MODE=true pytest tests/test_schemas.py -v     # 数据结构
MOCK_MODE=true pytest tests/test_exporters.py -v   # 导出器
MOCK_MODE=true pytest tests/test_pipeline.py -v    # 管线 + Tracker
```

---

## 九、常见问题

**Q: 没有 GPU 能跑吗？**
A: 可以。设置 `MOCK_MODE=true` 即可用模拟数据跑通全流程。导出格式验证完全不需要 GPU。

**Q: 视频处理很慢怎么办？**
A: 设置 `TTV_MAX_VIDEO_FRAMES=60` 限制最大拆帧数，或设置 `TTV_FRAME_SAMPLE_INTERVAL=1.0` 每秒采样一帧。

**Q: 如何查看中间层 JSON？**
A: 选择 `full_json` 导出格式即可得到完整的结构化 JSON。

**Q: 模型下载失败怎么办？**
A: 系统会自动降级到 mock 模式，不影响服务运行。手动下载权重文件到 `TTV_MODEL_CACHE_DIR` 即可。
