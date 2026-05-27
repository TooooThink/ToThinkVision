# ToThinkVision v2.1 — 视频 → 可编辑 4D 场景分解引擎

输入任意图片/视频（含 AI 生成视频），自动调用 **11 个顶级 AI 模型** 并行分析，输出 2D + 3D + **4D 时变场景** 全结构化数据，支持 **25+ 种导出格式**（含游戏引擎 3D 网格+纹理 / PSD 分层 / AE 动画 / .splat 高斯场 / **动画 glTF·USD·Blender 场景**）。

v2.1 新增 **4D 场景分解能力**：输入一段视频，输出每个物体的 3D 几何 + **6DoF 运动轨迹** + 动态场景图，可直接导入 Unity/Unreal/Blender/AE 进行编辑。

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

# 2. 安装 v2 基础模型（SAM 3 / Depth Pro / MASt3R 等）
chmod +x install_models.sh && ./install_models.sh

# 3. [可选] 安装 v2.1 进阶 4D 模型（CoTracker3 / ObjectGS / Spann3R / Shape of Motion）
chmod +x install_models_v3.sh && ./install_models_v3.sh
#   4 个模型都有 mock 回退，不装也能跑

# 4. 启动服务
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

> **提示：** 未找到模型权重时自动降级到 mock 模式，不会报错。

---

## 二、流水线：从视频到可编辑 4D 场景

```
输入：图片 / 视频 (.mp4 / .png) — 含 AI 生成的世界模型输出
  │
  ▼
┌──────────────────────────────────────────────────────────┐
│ 阶段1：感知（Perception）                                 │
│   SAM 3 + OmniParser + Grounding DINO + StrongSORT + OCR │
│   输出：每帧中每个物体的 2D bbox + mask + 跨帧track_id    │
├──────────────────────────────────────────────────────────┤
│ 阶段2：深度 + 2D 补全                                     │
│   Depth Pro 度量深度；LaMa 补全残缺物体                      │
│   输出：深度图 (H×W, 米) + 补全 mask                      │
├──────────────────────────────────────────────────────────┤
│ 阶段3：3D重建（Spann3R / MASt3R / VGGT）                  │
│   多视角融合 → 场景点云 → 逐对象网格 → UV展开 → 纹理烘焙   │
│   输出：3D点云 + 相机位姿 + 带贴图的3D网格 (.obj/.gltf)   │
├──────────────────────────────────────────────────────────┤
│ 阶段4：场景理解（Scene Understanding）                     │
│   地面平面检测 (RANSAC) → 重力对齐 → 物体分类             │
├──────────────────────────────────────────────────────────┤
│ 阶段5：3D 高斯 / ObjectGS（可选）                         │
│   场景级 3DGS 或 逐物体 3DGS                                │
├──────────────────────────────────────────────────────────┤
│ 阶段6：4D 轨迹提取 ⭐                                     │
│   方案A: Shape of Motion（端到端 4D 重建）                   │
│   方案B: ICP + PCA + B-spline + CoTracker3 增强            │
│   输出：逐物体 6DoF（位置+旋转+缩放+速度）                │
├──────────────────────────────────────────────────────────┤
│ 阶段7：4D 高斯泼溅（HexPlane，可选）⭐                     │
│   时变高斯场景表示                                           │
├──────────────────────────────────────────────────────────┤
│ 阶段8：动态场景图 ⭐                                       │
│   时变空间关系（above/near/contact）+ 交互事件               │
│   （碰撞、拾起、放下、接触开始/结束）                        │
├──────────────────────────────────────────────────────────┤
│ 阶段9：动画导出 ⭐                                         │
│   动画 glTF (.glb) / USDA / Blender 导入脚本                │
│   + 4D 场景图 JSON + AE 关键帧                              │
└──────────────────────────────────────────────────────────┘
```

---

## 三、核心模型（v2 全部默认开启）

| 模型 | 来源 | 显存 | 作用 |
|------|------|------|------|
| **SAM 3** | Meta AI | 12-24GB | 统一检测 + 分割 + 追踪（promptable concept） |
| **OmniParser v2** | Microsoft | 8-12GB | UI 元素精准检测 + 交互性预测 |
| **Grounding DINO** | IDEA-Research | 4-8GB | 开放词汇通用目标检测 |
| **StrongSORT** | CVPR 2023 | CPU | ReID + Kalman + GMC 多目标追踪 |
| **Depth Pro** | Apple (ICLR 2025) | 4-8GB | 真实米级深度估计，<1 秒推理 |
| **MASt3R / VGGT** | NAVER / Meta | 24-48GB | 视频 → 3D 点云 + 相机位姿 (SfM) |
| **Spann3R** ⭐ | 3DV 2025 | 24-48GB | 带空间记忆的 3D 重建（长序列抗漂移） |
| **ObjectGS** ⭐ | ICCV 2025 | 24GB | 逐物体 3D 高斯泼溅 |
| **CoTracker3** ⭐ | Meta AI | 8-12GB | 稠密点追踪 (265×265)，提升轨迹精度 |
| **Shape of Motion** ⭐ | ICCV 2025 | 24GB | 端到端单目视频 4D 重建 |
| **3D Gaussian Splatting** | Nerfstudio | 24GB | 视频 → 照片级 3D 高斯场景 |

⭐ v2.1 新增进阶 3D/4D 模型。所有模型可通过环境变量或 API 表单参数单独关闭，未安装权重/仓库时自动降级到 mock 模式。

---

## 四、你能获得什么

### 逐对象数据

- **2D**：bbox、分割 mask、轮廓、裁剪图
- **3D**：深度值、3D 位置 (bbox_3d)、点云索引
- **网格**：三角网格（顶点/面/法线）、UV 坐标、烘焙纹理（512×512 PNG）、逐对象 OBJ 文件
- **时序**：出现/消失帧、运动轨迹 [{x, y, t}]、速度、逐帧深度
- **外观**：主色、色板
- **关系**：与其他物体碰撞、相对方位（上/下/左/右）
- **交互**：可点击、可滚动、开关状态

### 全局场景数据

- **点云**：(N, 3) XYZ 点 + (N, 3) RGB 颜色 + (N, 3) 法线
- **相机位姿**：每帧内参 (3×3 K) + 外参 (4×4 RT) + 位置 + 四元数旋转
- **高斯参数**：means, quaternions, scales, opacities, 球谐系数
- **场景网格**：合并 OBJ 文件，含所有对象网格、UV、纹理、MTL 材质

---

## 五、25+ 种导出格式

### 3D 网格与场景
| 格式 | 扩展名 | 说明 |
|------|--------|------|
| `gltf` | `.gltf` + `.bin` | glTF 2.0，含 UV 坐标、嵌入纹理、PBR 材质、相机位姿节点 |
| `obj_3d` | `.obj` + `.mtl` | Wavefront OBJ，含 UV、MTL 材质、纹理引用 (map_Kd)、相机位姿注释 |

### 游戏引擎
| 格式 | 扩展名 | 说明 |
|------|--------|------|
| `unity_json` | `.json` | Unity 场景，含 3D 网格 (MeshFilter + MeshRenderer + MeshCollider)、纹理、变换 |
| `ue_json` | `.json` | UE5 Actor，含 StaticMeshComponent、MaterialInterface、变换 |
| `unity_splat` | `.splat` | 3D 高斯泼溅二进制，用于 UnityGaussianSplatting 插件 |
| `ue_splat` | `.splat` | 3D 高斯泼溅二进制，用于 UnrealSplat 插件 |
| `collision_json` | `.json` | 纯碰撞盒数据，用于物理引擎 |

### 动画与设计
| 格式 | 扩展名 | 说明 |
|------|--------|------|
| `psd_static` | `.psd` | Photoshop PSD — 每个物体一个透明图层 |
| `psd_animated` | `.psd` | Photoshop PSD — 每帧一个 Group，含对象图层 |
| `ae_project` | `.jsx` + `.json` | After Effects ExtendScript — 自动创建含跟踪图层+摄像机的合成 |
| `ae_keyframes` | `.json` | AE 关键帧时间线（位置/缩放/旋转/透明度） |
| `video_trajectory` | `.csv` | 逐帧物体轨迹 CSV |
| `pr_markers` | `.json` | Premiere Pro 章节标记，含时间码 |

### UI 与具身智能
| 格式 | 扩展名 | 说明 |
|------|--------|------|
| `figma_json` | `.json` | Figma 文档结构（RECTANGLE/TEXT/FRAME 节点） |
| `html_css` | `.html` | 自包含 HTML，绝对定位元素 |
| `ui_json` | `.json` | 简化 UI 组件列表 |
| `embodied_json` | `.json` | 场景对象含 3D 位姿 + 物理属性 + 交互序列 |
| `robot_action` | `.json` | 机器人接近/抓取动作序列 |
| `pose_csv` | `.csv` | 逐帧 3D 位姿 CSV |

### 4D 场景（v2.1 新增）⭐
| 格式 | 扩展名 | 说明 |
|------|--------|------|
| `animated_gltf` | `.glb` | 动画 glTF 二进制 — 逐物体 3D 网格 + 关键帧动画（位移+旋转+缩放），Unity/Blender 直接导入 |
| `usd_scene` | `.usda` | USD 文本格式，含时间采样的变换、材质、动画摄像机，可用于 Unreal/Omniverse |
| `blender_scene` | `.py` + `.obj` + `.json` | Blender Python 导入脚本 + 逐物体 OBJ 网格 + 轨迹 JSON |
| `scene_graph_json` | `.json` | 动态 4D 场景图：节点（物体）+ 边（时变关系）+ 交互事件 |

### 通用
| 格式 | 扩展名 | 说明 |
|------|--------|------|
| `full_json` | `.json` | 完整结构化输出，含所有 2D + 3D + 4D + 网格 + 相机 + 高斯数据 |

---

## 六、代码结构详解

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
│   │   ├── StructuredObject         → 物体: bbox/mask/3d/depth/color/text/trajectory/mesh_3d
│   │   ├── StructuredOutput         → 完整输出 + PointCloud + CameraPose + GaussianSplat + Mesh3D
│   │   ├── Mesh3D                   → 3D 三角网格 (vertices/faces/normals/uv/texture)
│   │   ├── PointCloud               → 3D 点云 (points/colors/normals/confidence)
│   │   ├── CameraPose               → 相机位姿 (intrinsics/extrinsics/quaternion)
│   │   ├── GaussianSplatData        → 高斯参数 (means/quats/scales/opacities/SH)
│   │   ├── PSDLayer                 → PSD 图层结构
│   │   ├── ObjectType               → 物体类型枚举（5 大类）
│   │   └── ExportFormat             → 20+ 种导出格式枚举
│   │
│   ├── preprocessor.py               # 文件预处理（类型判断/拆帧/清理）
│   ├── pipeline.py                   # 主流程编排（图片/视频管线）★
│   │
│   ├── models/                       # AI 模型封装（均有 mock 回退）
│   │   ├── sam3.py                  → SAM 3: 检测+分割+追踪（替换 segmentor.py）
│   │   ├── omniparser.py            → OmniParser v2: UI 元素检测
│   │   ├── grounding_dino.py        → Grounding DINO: 开放词汇检测
│   │   ├── strongsort_wrapper.py    → StrongSORT: 多目标追踪
│   │   ├── depth_pro.py             → Depth Pro: 度量深度估计
│   │   ├── mast3r.py                → MASt3R/VGGT: 3D 点云 + 相机位姿
│   │   ├── gaussian_splatting.py    → 3DGS: 训练 + .splat/.ply 导出
│   │   ├── mesh_reconstruction.py   → ★ 逐对象3D网格: 深度反投影→Poisson→UV→纹理
│   │   ├── cotracker3.py            → ★ CoTracker3: 稠密点追踪 (265×265)
│   │   ├── object_gs.py             → ★ ObjectGS: 逐物体 3D 高斯泼溅
│   │   ├── spann3r.py               → ★ Spann3R: 带空间记忆的 3D 重建
│   │   ├── shape_of_motion.py       → ★ Shape of Motion: 端到端 4D 重建
│   │   ├── trajectory_4d.py         → ★ 6DoF 轨迹: ICP + PCA + B-spline
│   │   └── gaussian_splatting_4d.py → ★ 4DGS (HexPlane 时间分解)
│   │
│   ├── scene/                       # 4D 场景理解
│   │   ├── scene_graph_4d.py        → ★ 动态场景图构建器
│   │   └── world_model_adapter.py   → ★ AI 生成视频检测 + 阈值调节
│   │
│   ├── exporters/                    # 导出层
│   │   ├── base.py                  → Exporter 基类
│   │   ├── gltf_exporter.py         → glTF 2.0: UV + 纹理 + PBR 材质
│   │   ├── obj_exporter.py          → Wavefront OBJ + MTL + UV + 纹理
│   │   ├── game_exporter.py         → Game 3D: Unity/UE/Collision (3D网格感知)
│   │   ├── video_exporter.py        → Video: AE/Trajectory/PR
│   │   ├── embodied_exporter.py     → Embodied: Robot/Pose/Action
│   │   ├── psd_exporter.py          → PSD: 静态分层 + 视频动画分层
│   │   ├── ae_project_exporter.py   → AE: ExtendScript .jsx + 关键帧动画
│   │   ├── splat_exporter.py        → Splat: .splat 二进制 + .ply 高斯参数
│   │   ├── ui_exporter.py           → UI: Figma/HTML/UI JSON
│   │   ├── image_exporter.py        → 裁剪图 / mask / 深度可视化
│   │   ├── animated_gltf_exporter.py → ★ 动画 .glb (逐物体关键帧动画)
│   │   ├── usd_exporter.py          → ★ USDA (时间采样变换)
│   │   ├── blender_exporter.py      → ★ Blender Python 脚本 + OBJ + 轨迹 JSON
│   │   └── manifest.py              → 导出清单（README.txt）
│   │
│   └── utils/
│       ├── camera.py                → 相机工具: 内参/外参/坐标系变换/四元数
│       ├── pointcloud.py            → 点云处理: 反投影/法线/体素滤波/PLY
│       ├── color.py                 → 主色提取
│       ├── geometry.py              → 碰撞检测/相对方位/z-index
│       ├── texture_bake.py          → UV展开 + 多视角纹理烘焙 ★
│       ├── scene_understanding.py   → 地面检测(RANSAC)/重力对齐/场景分类 ★
│       └── io.py                    → JSON 读写
│
├── docs/
│   └── GETTING_STARTED.md           # ★ 新手入门：知识图谱 + 学习路径
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
├── install_models.sh                 # v2 基础模型交互式下载脚本
├── install_models_v2.sh              # v2 进阶模型下载脚本
├── install_models_v3.sh              # v2.1 4D 模型交互式下载脚本（CoTracker3/ObjectGS/Spann3R/Shape of Motion）
├── README.md                         # 英文文档
└── README_CN.md                      # ← 你正在看的中文文档
```

---

## 七、如何使用

### Web 界面

1. 启动服务后打开 `http://localhost:8000`
2. 拖拽上传图片/视频
3. 选择导出格式（可多选）
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

# 上传视频 + 导出 glTF 3D 场景（含UV+纹理）
curl -X POST http://localhost:8000/api/process \
  -F "file=@demo.mp4" \
  -F 'export_formats=["gltf", "unity_json"]' \
  -F "mode=general"

# 下载结果
curl -O http://localhost:8000/api/download/demo.gltf
```

### Python 直接调用

```python
from app.pipeline import process_file
from app.exporters.gltf_exporter import GltfExporter
from app.schemas import ExportFormat

# 处理视频（自动调用 SAM 3 + MASt3R + 网格重建）
result = process_file("demo.mp4", mode="video")

# 查看 3D 数据
print(f"点云: {len(result.point_cloud.points)} 点")
print(f"相机位姿: {len(result.camera_poses)} 帧")
print(f"有3D网格的对象: {sum(1 for o in result.objects if o.mesh_3d)}")
print(f"场景网格文件: {result.scene_mesh_path}")

# 导出 glTF（含UV+纹理+PBR材质）
exporter = GltfExporter()
output_path = exporter.export(result)
```

---

## 八、中间层 JSON 数据结构（v2 扩展）

### 物体级别（新增 3D 网格 + 纹理）

```json
{
  "id": "obj_0001",
  "label": "game_item",
  "label_custom": "桌子",
  "confidence": 0.92,

  "bbox": { "x": 100, "y": 200, "w": 120, "h": 40 },
  "mask_base64": "iVBORw0KGgo...",
  "crop_image_base64": "iVBORw0KGgo...",
  "crop_png_path": "outputs/demo/obj_0001_crop.png",

  "bbox_3d": { "x": 160, "y": 220, "z": 2.5 },
  "depth_value": 2.5,

  "mesh_3d": {
    "vertices": [[x, y, z], ...],
    "faces": [[0, 1, 2], ...],
    "normals": [[nx, ny, nz], ...],
    "uv_coords": [[u, v], ...],
    "uv_face_map": [[0, 1, 2], ...],
    "texture_path": "outputs/demo/meshes/obj_0001_texture.png",
    "texture_base64": "iVBORw0KGgo...",
    "bounds": {"min": [x, y, z], "max": [x, y, z]},
    "point_count": 5230
  },
  "mesh_obj_file": "outputs/demo/meshes/obj_0001.obj",

  "dominant_color": "#8b6914",
  "z_index": 3,

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
  },
  "scene_mesh_path": "outputs/demo/meshes/scene_mesh.obj"
}
```

---

## 九、环境变量

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
| `TTV_ENABLE_MAST3R` | `true` | 是否启用 MASt3R / VGGT |
| `TTV_ENABLE_GAUSSIAN_SPLATTING` | `false` | 是否启用 3DGS 训练 |
| **v2.1 进阶模型**（需单独执行 `install_models_v3.sh`） | | |
| `TTV_ENABLE_COTRACKER3` | `true` | 是否启用 CoTracker3 稠密点追踪 |
| `TTV_ENABLE_OBJECTGS` | `false` | 是否启用 ObjectGS（需 clone 仓库） |
| `TTV_ENABLE_SPANN3R` | `false` | 是否启用 Spann3R（需 clone 仓库） |
| `TTV_ENABLE_SHAPE_OF_MOTION` | `false` | 是否启用 Shape of Motion（需 clone 仓库） |
| `TTV_ENABLE_4D_TRAJECTORY` | `true` | 是否启用 4D 6DoF 轨迹提取 |
| `TTV_ENABLE_4DGS` | `false` | 是否启用 4D 高斯泼溅（重量级，多卡） |
| `TTV_ENABLE_SCENE_GRAPH` | `true` | 是否启用动态场景图构建 |
| `TTV_ENABLE_ANIMATED_EXPORT` | `true` | 是否启用动画 glTF/USD/Blender 导出 |
| `OBJECT_GS_PATH` | — | ObjectGS 仓库 clone 路径 |
| `SPANN3R_PATH` | — | Spann3R 仓库 clone 路径 |
| `SHAPE_OF_MOTION_PATH` | — | Shape of Motion 仓库 clone 路径 |
| `TTV_OUTPUT_DIR` | `./outputs` | 导出文件输出目录 |

---

## 十、运行测试

```bash
# 运行全部测试（112+ 个用例）
MOCK_MODE=true pytest tests/ -v

# 按模块运行
MOCK_MODE=true pytest tests/test_schemas.py -v      # 数据结构 + 导出器 + 工具
MOCK_MODE=true pytest tests/test_exporters.py -v    # 导出器格式验证
MOCK_MODE=true pytest tests/test_pipeline.py -v     # Tracker + Geometry + Color
MOCK_MODE=true pytest tests/test_4d_scene.py -v     # ★ 4D 轨迹 + 动画导出 + 场景图 (63)
```

---

## 十一、新手入门

刚接触计算机视觉 / 3D 重建？阅读 **[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)** — 完整的新手指南，涵盖：

- 20 个核心知识点，分 6 层递进
- 每个知识点在项目代码中的对应位置
- 推荐学习路径（8 个阶段 + 实践任务）
- 5 个最容易出研究突破的方向
- 课程、论文、工具资源链接

---

## 十二、如何进一步开发

### 12.1 添加新模型

在 `app/models/` 下新建文件实现相同接口，在 `pipeline.py` 中调用即可。所有模型封装都自带 mock 回退。

```python
# app/models/yolo_detector.py
def detect_objects(img, mode="general"):
    # 实现
    return [{"bbox": [...], "label": "xxx", "confidence": 0.9}]
```

### 12.2 添加新导出格式

1. 在 `app/schemas.py` 的 `ExportFormat` 枚举中添加
2. 在 `app/exporters/` 下新建导出器，继承 `BaseExporter`
3. 在 `app/main.py` 的 `EXPORT_FORMAT_MAP` 中注册
4. 在 `static/index.html` 的 `FORMAT_MAP` 中添加前端选项

### 12.3 优化 3D 重建

- 集成 COLMAP 做更精确的相机位姿估计
- 用 nerfstudio 替代手写 3DGS 训练循环
- 从 NeRF/3DGS 密度场提取高质量网格（替代深度反投影）
- 用 VLM（如 Qwen-VL）辅助语义分割和场景理解

### 12.4 增强 PSD/AE 导出

- 使用 PhotoshopAPI 写入真实 PSD 文件
- 在 AE 脚本中添加遮罩/形状图层
- 导出 AEPX 工程文件（需要 AE 脚本解析器）

### 12.5 批量处理 & 异步任务

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

### 12.6 部署

```bash
# Docker
docker build -t tothinkvision .
docker run -p 8000:8000 --gpus all tothinkvision

# Nginx 反代
# server { listen 80; location / { proxy_pass http://127.0.0.1:8000; } }
```

---

## 十三、常见问题

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

**Q: 3D 网格怎么导入 Unity/Blender？**
A: glTF: 直接拖入 Blender，或用 [gltf-viewer](https://gltf-viewer.donmccurdy.com) 在浏览器中预览。OBJ: 通用格式，所有 3D 软件都能导入。Unity JSON: 需要配套 C# 导入脚本（详见 manifest 说明）。

**Q: 怎么安装 v2.1 新增的 4 个模型（CoTracker3 / ObjectGS / Spann3R / Shape of Motion）？**
A: 运行 `chmod +x install_models_v3.sh && ./install_models_v3.sh`，交互式选择安装。CoTracker3 只需下载权重（torch.hub 自动下载 ~300MB）；ObjectGS / Spann3R / Shape of Motion 需要 clone 仓库 + 安装子模块 + 各自依赖（共约 18GB）。4 个模型全部有 mock 回退，不装也能跑完整流程。

**Q: 动画 glTF / USD / Blender 导出怎么用？**
A: **动画 glTF (.glb)**：直接拖入 Blender 或 [gltf-viewer](https://gltf-viewer.donmccurdy.com) 在线预览；Unity 通过 `GameObject > Import Package` 导入，Unreal 通过 glTF 插件导入。**USDA**：用 NVIDIA Omniverse 打开，或通过 USD 插件导入 Unreal。**Blender 导出**：在 Blender 中运行生成的 `.py` 脚本（`File > Open > scene.py`），会自动导入所有逐物体 OBJ 网格并应用关键帧动画。

**Q: 能处理 AI 生成的视频（Sora/Veo/Kling 等）吗？**
A: 能。勾选 "World Model Video" 选项（或 API 设置 `is_world_model_video=True`）。流水线会自动：(1) 通过 `WorldModelAdapter` 检测 AI 视频特征；(2) 放宽 ICP/形变阈值（2×/1.5×）；(3) 增加 B-spline 平滑（+0.2~0.3）吸收时间抖动。推荐后端用 Shape of Motion，端到端、对非物理几何更鲁棒。

**Q: 4D Trajectory 和 Shape of Motion 什么区别？**
A: **4D Trajectory**（默认开，CPU）是流水线方案：深度图 + mask + ICP 对齐 → 逐物体 6DoF，可配合任何深度估计器。**Shape of Motion**（可选，24GB 显存）是端到端单目视频 4D 重建，联合优化几何与运动，不需要单独的深度/追踪步骤。显存够优先用 Shape of Motion，显存紧张退回到 4D Trajectory。
