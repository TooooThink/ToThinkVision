# ToThinkVision v2.0 — 四合一通用视觉结构化引擎

输入任意图片/视频，自动调用 **7 个顶级 AI 模型** 并行分析，输出 2D + 3D 全结构化数据，支持 **20+ 种导出格式**（含游戏引擎 3D / PSD 分层 / AE 动画 / .splat 高斯场）。

开源研究 & 工业项目，默认全部开启最强配置，用户可自由选择关闭某些模型。

---

## 一、快速上手

### 最快体验（无需 GPU / 模型）

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动服务（mock 模式，用模拟数据跑通全流程）
MOCK_MODE=true uvicorn app.main:app --host 0.0.0.0 --port 8000

# 3. 浏览器打开 http://localhost:8000
```

### 完整部署（使用真实 AI 模型）

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 交互式安装脚本（可选择安装单个或多个模型）
chmod +x install_models.sh && ./install_models.sh

# 3. 启动服务
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

> **提示：** 未找到模型权重时自动降级到 mock 模式，不会报错。

---

## 二、核心模型（v2 全部默认开启）

| 模型 | 来源 | 显存 | 作用 |
|------|------|------|------|
| **SAM 3** | Meta AI | 12-24GB | 统一检测 + 分割 + 追踪（promptable concept） |
| **OmniParser v2** | Microsoft | 8-12GB | UI 元素精准检测 + 交互性预测 |
| **Grounding DINO 1.6** | IDEA-Research | 4-8GB | 开放词汇通用目标检测 |
| **StrongSORT** | CVPR 2023 | CPU | ReID + Kalman + GMC 多目标追踪 |
| **Depth Pro** | Apple (ICLR 2025) | 4-8GB | 真实米级深度估计，<1 秒推理 |
| **MASt3R** | Meta (2025) | 24-48GB | 视频 → 3D 点云 + 相机位姿 |
| **3D Gaussian Splatting** | Nerfstudio | 24GB | 视频 → 照片级 3D 高斯场景 |

所有模型均可通过环境变量单独关闭，也可全部启用获得最佳效果。

---

## 三、功能管线

```
输入层
  ├── 图片 → 预处理
  └── 视频 → FFmpeg 拆帧（可配置采样率）

分析层（全模型并行，默认全部开启）
  ├── SAM 3 → 检测 + 分割 + 追踪（三合一）
  ├── OmniParser v2 → UI 模式精准元素检测
  ├── Grounding DINO 1.6 → 通用模式开放词汇检测
  ├── Depth Pro → 每帧度量深度（米）
  └── StrongSORT → 跨帧追踪（兜底）

3D 重建层
  ├── MASt3R → 采样帧对 → 3D 点云 + 相机位姿
  └── 3DGS (gsplat) → 训练高斯场 → .splat/.ply

中间层 JSON（扩展版）
  ├── StructuredObject: id, bbox, mask, 3d_pose, depth_meters, color, text, trajectory, relations
  ├── PointCloud: points(N,3), colors(N,3), normals(N,3), confidence(N)
  ├── CameraPose: frame_idx, intrinsics(3x3), extrinsics(4x4), quaternion
  └── GaussianSplat: means, quats, scales, opacities, SH_coeffs

导出层（20+ 种格式）
  ├── UI: Figma JSON, HTML+CSS, UI JSON
  ├── Game 3D: Unity .splat + 导入脚本, UE5 .splat, glTF/PLY, OBJ, Unity/UE JSON
  ├── Video: AE 关键帧 JSON, 轨迹 CSV, PR 标记
  ├── PSD + AEPX: Photoshop PSD (分层) + After Effects 工程 (动画)
  ├── .splat 二进制: UnityGaussianSplatting / UnrealSplat 插件直接加载
  └── Embodied: Robot Actions, Pose CSV
```

---

## 四、20+ 种导出格式

### UI 解构
- `figma_json` — Figma 文档结构
- `html_css` — 带内联 CSS 的自包含 HTML
- `ui_json` — 简化 UI 组件 JSON

### 游戏 3D
- `unity_json` — Unity GameObject + Collider + Rigidbody + 3D 点云摘要
- `ue_json` — Unreal Engine Actor 结构
- `collision_json` — 纯碰撞盒数据
- `unity_splat` — .splat 二进制 + Unity C# 导入脚本
- `ue_splat` — .splat 二进制 + UE 导入说明
- `gltf` — .ply 格式（含高斯参数，可转 glTF）
- `obj_3d` — OBJ 网格文件

### PSD + AE 动画
- `psd_static` — Photoshop PSD 分层文件（每物体 = 一层）
- `psd_animated` — Photoshop PSD 分层文件（每帧 = Group）
- `ae_project` — After Effects ExtendScript (.jsx) + 数据 JSON

### 视频动效
- `ae_keyframes` — After Effects 关键帧时间线 JSON
- `video_trajectory` — 逐帧物体轨迹 CSV
- `pr_markers` — Premiere Pro 章节标记

### 具身智能
- `embodied_json` — 场景 + 交互序列
- `robot_action` — 机器人接近/抓取动作序列
- `pose_csv` — 逐帧 3D 位姿 CSV

### 通用
- `full_json` — 完整中间层结构化 JSON（含 3D 数据）

---

## 五、代码结构详解

```
ToThinkVision/
├── app/
│   ├── main.py                       # FastAPI 服务 + REST API
│   │   ├── GET  /                   → 前端页面
│   │   ├── GET  /api/formats        → 所有导出格式列表
│   │   ├── POST /api/process        → 上传 + 处理 + 导出
│   │   ├── GET  /api/download       → 下载文件
│   │   └── GET  /api/health         → 健康检查 + 模型版本
│   │
│   ├── config.py                     # 全局配置（环境变量）
│   │   └── 全部模型默认开启，可通过 TTV_ENABLE_* 关闭
│   │
│   ├── schemas.py                    # 统一中间层 Pydantic 模型
│   │   ├── StructuredObject         → 物体: bbox/mask/3d/depth/color/text/trajectory
│   │   ├── StructuredOutput         → 完整输出 + PointCloud + CameraPose + GaussianSplat
│   │   ├── PointCloud               → 3D 点云 (points/colors/normals/confidence)
│   │   ├── CameraPose               → 相机位姿 (intrinsics/extrinsics/quaternion)
│   │   ├── GaussianSplatData        → 高斯参数 (means/quats/scales/opacities/SH)
│   │   ├── PSDLayer                 → PSD 图层结构
│   │   ├── ObjectType               → 物体类型枚举（5 大类）
│   │   └── ExportFormat             → 20+ 种导出格式枚举
│   │
│   ├── preprocessor.py               # 文件预处理（类型判断/拆帧/清理）
│   ├── pipeline.py                   # 主流程编排（图片/视频管线）
│   │
│   ├── models/                       # AI 模型封装（均有 mock 回退）
│   │   ├── sam3.py                  → SAM 3: 检测+分割+追踪（替换 segmentor.py）
│   │   ├── omniparser.py            → OmniParser v2: UI 元素检测
│   │   ├── grounding_dino.py        → Grounding DINO 1.6: 开放词汇检测
│   │   ├── strongsort_wrapper.py    → StrongSORT: 多目标追踪
│   │   ├── depth_pro.py             → Depth Pro: 度量深度估计
│   │   ├── mast3r.py                → MASt3R: 3D 点云重建 + 相机位姿
│   │   └── gaussian_splatting.py    → 3DGS: 训练 + .splat/.ply 导出
│   │
│   ├── exporters/                    # 导出层
│   │   ├── base.py                  → Exporter 基类
│   │   ├── ui_exporter.py           → UI: Figma/HTML/UI JSON
│   │   ├── game_exporter.py         → Game 3D: Unity/UE/Collision + 3D 点云
│   │   ├── video_exporter.py        → Video: AE/Trajectory/PR
│   │   ├── embodied_exporter.py     → Embodied: Robot/Pose/Action
│   │   ├── psd_exporter.py          → PSD: 静态分层 + 视频动画分层
│   │   ├── ae_project_exporter.py   → AE: ExtendScript .jsx + 关键帧动画
│   │   └── splat_exporter.py        → Splat: .splat 二进制 + .ply 高斯参数
│   │
│   └── utils/
│       ├── pointcloud.py            → 点云处理: back-projection/normals/filtering/PLY
│       ├── camera.py                → 相机工具: intrinsics/extrinsics/quaternion
│       ├── color.py                 → 主色提取
│       ├── geometry.py              → 碰撞检测/相对方位/z-index
│       └── io.py                    → JSON 读写
│
├── static/
│   ├── index.html                   → 单页 Web UI
│   └── style.css                    → 暗色主题样式
│
├── tests/
│   ├── test_schemas.py              → v2 数据结构 + 导出器 + 工具（19 个）
│   ├── test_exporters.py            → 12 种导出格式验证（12 个）
│   └── test_pipeline.py             → Tracker + Geometry + Color + 端到端（20 个）
│
├── uploads/                          # 上传暂存（自动创建）
├── outputs/                          # 导出输出
├── requirements.txt                  # 依赖清单
├── install_models.sh                 # 交互式模型下载脚本
├── README.md                         # 英文文档
└── README_CN.md                      # ← 你正在看的中文文档
```

---

## 六、如何使用

### Web 界面

1. 启动服务后打开 `http://localhost:8000`
2. 拖拽上传图片/视频
3. 选择导出格式
4. 点击「Process & Export」
5. 下载结果

### API 调用

```bash
# 列出所有导出格式
curl http://localhost:8000/api/formats

# 上传图片 + 导出 PSD 分层
curl -X POST http://localhost:8000/api/process \
  -F "file=@screenshot.png" \
  -F "export_format=psd_static" \
  -F "mode=ui"

# 上传视频 + 导出 .splat 3D 高斯场景
curl -X POST http://localhost:8000/api/process \
  -F "file=@demo.mp4" \
  -F "export_format=unity_splat" \
  -F "mode=video"

# 下载结果
curl -O http://localhost:8000/api/download/demo_unity.splat
```

### Python 直接调用

```python
from app.pipeline import process_file
from app.exporters.splat_exporter import SplatExporter
from app.schemas import ExportFormat

# 处理视频（自动调用 SAM 3 + MASt3R + 3DGS）
result = process_file("demo.mp4", mode="video")

# 导出 .splat 高斯场景
exporter = SplatExporter(ExportFormat.UNITY_SPLAT)
output_path = exporter.export(result)

# 查看 3D 数据
print(f"点云: {len(result.point_cloud.points)} 点")
print(f"相机位姿: {len(result.camera_poses)} 帧")
print(f"高斯数量: {len(result.gaussian_splats.means)}")
```

---

## 七、中间层 JSON 数据结构（v2 扩展）

### 物体级别（新增 3D + 蒙版）

```json
{
  "id": "obj_0001",
  "label": "ui_button",
  "label_custom": "提交按钮",
  "confidence": 0.92,

  "bbox": { "x": 100, "y": 200, "w": 120, "h": 40 },
  "mask_base64": "iVBORw0KGgo...",
  "crop_image_base64": "iVBORw0KGgo...",
  "contour": [{ "x": 100, "y": 200 }, ...],

  "bbox_3d": { "x": 160, "y": 220, "z": 2.5 },
  "depth_value": 2.5,

  "dominant_color": "#3b82f6",
  "z_index": 5,

  "text_content": "Submit",

  "temporal": {
    "frame_index": 10,
    "appear_frame": 0,
    "disappear_frame": 89,
    "trajectory": [{ "x": 160, "y": 220, "t": 0 }, ...],
    "velocity": { "vx": 0.5, "vy": 0.2 },
    "depth_per_frame": [2.5, 2.6, 2.4, ...]
  },

  "relations": {
    "collision_with": ["obj_0002"],
    "relative_positions": [{ "target_id": "obj_0002", "relation": "above" }]
  },

  "point_cloud_indices": [0, 1, 2, ...]
}
```

### 全局输出（新增 3D 字段）

```json
{
  "source_file": "demo.mp4",
  "source_type": "video",
  "metadata": { "fps": 30, "width": 1920, "height": 1080, "duration_seconds": 3.0 },
  "objects": [ ... ],
  "point_cloud": {
    "points": [[x, y, z], ...],
    "colors": [[r, g, b], ...],
    "normals": [[nx, ny, nz], ...],
    "confidence": [0.9, 0.8, ...]
  },
  "camera_poses": [
    {
      "frame_idx": 0,
      "intrinsics": [[500,0,320],[0,500,240],[0,0,1]],
      "extrinsics": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
      "position": [0, 0, -2],
      "rotation": [1, 0, 0, 0]
    }
  ],
  "gaussian_splats": {
    "means": [[x, y, z], ...],
    "quats": [[w, x, y, z], ...],
    "scales": [[sx, sy, sz], ...],
    "opacities": [0.8, ...],
    "sh_coeffs": [[c0, c1, c2], ...]
  }
}
```

---

## 八、环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TTV_MOCK_MODE` | `false` | `true` 使用模拟数据（无 GPU 也能跑） |
| `TTV_DEVICE` | `cuda` | 推理设备：`cuda` / `cpu` / `mps` |
| `TTV_MAX_VIDEO_FRAMES` | `300` | 视频最大拆帧数 |
| `TTV_MODEL_CACHE_DIR` | `~/.cache/tothinkvision` | 模型权重路径 |
| `TTV_MAX_UPLOAD_MB` | `500` | 最大上传文件大小（MB） |
| `TTV_ENABLE_SAM3` | `true` | 是否启用 SAM 3 |
| `TTV_ENABLE_OMNIPARSER` | `true` | 是否启用 OmniParser v2 |
| `TTV_ENABLE_GROUNDING_DINO` | `true` | 是否启用 Grounding DINO |
| `TTV_ENABLE_STRONGSORT` | `true` | 是否启用 StrongSORT |
| `TTV_ENABLE_DEPTH_PRO` | `true` | 是否启用 Depth Pro |
| `TTV_ENABLE_MAST3R` | `true` | 是否启用 MASt3R |
| `TTV_ENABLE_GAUSSIAN_SPLATTING` | `true` | 是否启用 3DGS 训练 |
| `TTV_OUTPUT_DIR` | `./outputs` | 导出文件输出目录 |

---

## 九、运行测试

```bash
# 运行全部测试（51 个用例）
MOCK_MODE=true pytest tests/ -v

# 按模块运行
MOCK_MODE=true pytest tests/test_schemas.py -v      # 数据结构 + 导出器 + 工具
MOCK_MODE=true pytest tests/test_exporters.py -v    # 导出器格式验证
MOCK_MODE=true pytest tests/test_pipeline.py -v     # Tracker + Geometry + Color
```

---

## 十、如何进一步开发

### 10.1 添加新模型

在 `app/models/` 下新建文件实现相同接口，在 `pipeline.py` 中调用即可。所有模型封装都自带 mock 回退。

```python
# app/models/yolo_detector.py
def detect_objects(img, mode="general"):
    # 实现
    return [{"bbox": [...], "label": "xxx", "confidence": 0.9}]
```

### 10.2 添加新导出格式

1. 在 `app/schemas.py` 的 `ExportFormat` 枚举中添加
2. 在 `app/exporters/` 下新建导出器，继承 `BaseExporter`
3. 在 `app/main.py` 的 `EXPORT_FORMAT_MAP` 中注册
4. 在 `static/index.html` 的 `FORMAT_MAP` 中添加前端选项

### 10.3 优化 3D 重建

- 替换 MASt3R 为其他 SfM 方案（COLMAP + NeRF）
- 调整 3DGS 训练参数（迭代次数、SH 阶数）
- 添加网格重建（Poisson / Marching Cubes）

### 10.4 增强 PSD/AE 导出

- 使用 PhotoshopAPI 写入真实 PSD 文件
- 在 AE 脚本中添加遮罩/形状图层
- 导出 AEPX 工程文件（需要 AE 脚本解析器）

### 10.5 批量处理 & 异步任务

```python
# 使用 Celery 或 FastAPI BackgroundTasks
@app.post("/api/batch_process")
async def batch_process(files: list[UploadFile] = File(...)):
    results = []
    for f in files:
        result = process_file(...)
        results.append(result)
    return results
```

### 10.6 部署

```bash
# Docker
docker build -t tothinkvision .
docker run -p 8000:8000 --gpus all tothinkvision

# Nginx 反代
# server { listen 80; location / { proxy_pass http://127.0.0.1:8000; } }
```

---

## 十一、常见问题

**Q: 没有 GPU 能跑吗？**
A: 可以。设置 `MOCK_MODE=true` 用模拟数据跑通全流程。导出格式验证不需要 GPU。

**Q: 显存不够怎么办？**
A: 可通过环境变量关闭部分模型，如 `TTV_ENABLE_MAST3R=false` `TTV_ENABLE_GAUSSIAN_SPLATTING=false`，只保留轻量模型运行。

**Q: 视频处理很慢？**
A: 设置 `TTV_MAX_VIDEO_FRAMES=60` 限制拆帧数，或 `TTV_FRAME_SAMPLE_INTERVAL=1.0` 每秒采一帧。

**Q: 模型下载失败？**
A: 自动降级到 mock 模式，不影响服务运行。手动下载权重到 `TTV_MODEL_CACHE_DIR` 即可。

**Q: .splat 文件怎么用？**
A: Unity: 导入 [UnityGaussianSplatting](https://github.com/aras-p/UnityGaussianSplatting) 插件，将 `.splat` 拖入场景即可实时渲染。UE: 使用 [UnrealSplat](https://github.com/mrquicksilver/UnrealSplat) 插件。
