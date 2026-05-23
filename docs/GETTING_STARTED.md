# ToThinkVision 入门指南

> 从输入视频到游戏引擎/AE可编辑的3D场景 — 你需要知道的一切。

---

## 目录

- [一、项目是什么](#一项目是什么)
- [二、整体流水线：五个阶段](#二整体流水线五个阶段)
- [三、项目目录结构](#三项目目录结构)
- [四、需要学习的知识体系](#四需要学习的知识体系)
  - [第一层：基础（必须先学）](#第一层基础必须先学)
  - [第二层：感知（画面里有什么）](#第二层感知画面里有什么)
  - [第三层：深度与3D重建（真实世界长什么样）](#第三层深度与3d重建真实世界长什么样)
  - [第四层：场景理解（这是什么场景）](#第四层场景理解这是什么场景)
  - [第五层：导出（目标引擎能读懂什么）](#第五层导出目标引擎能读懂什么)
  - [第六层：进阶研究方向](#第六层进阶研究方向)
- [五、知识点之间的关系](#五知识点之间的关系)
- [六、推荐学习路径](#六推荐学习路径)
- [七、最容易出研究突破的方向](#七最容易出研究突破的方向)
- [八、常用资源与参考链接](#八常用资源与参考链接)

---

## 一、项目是什么

**ToThinkVision** 是一个计算机视觉管线项目。它的目标是：

> **输入图片/视频（实景拍摄或AI生成），自动解析出场景中每个物体的（3D）坐标、运动轨迹、分割mask、深度信息，并导出为PS、游戏引擎（Unity/UE5）、动画软件（After Effects/Photoshop）可以直接使用的格式。**
本文为其3D作用入门。

### 应用场景举例

| 场景 | 输入 | 输出 |
|------|------|------|
| 游戏开发 | 实景房间视频 | Unity场景，含3D家具模型+碰撞体 |
| 影视后期 | 实景镜头 | AE工程，含跟踪图层+摄像机反求 |
| 平面设计 | 产品照片 | PSD分层文件，每个物体独立图层 |
| 机器人 | 环境视频 | 3D场景JSON，含障碍物位置+物理属性 |

---

## 二、整体流水线：五个阶段

```
输入视频（.mp4 / .png）
  │
  ▼
┌──────────────────────────────────────────────────────────┐
│ 阶段1：感知（Perception）                                 │
│   "画面里有什么？在哪里？"                                  │
│                                                          │
│   目标检测 → 实例分割 → 多目标跟踪 → OCR文字识别            │
│   输出：每帧中每个物体的 2D bbox + mask + 跨帧track_id     │
├──────────────────────────────────────────────────────────┤
│ 阶段2：深度（Depth）                                      │
│   "每个像素离我多远？"                                      │
│                                                          │
│   单目深度估计（每帧独立）                                   │
│   输出：深度图 (H×W)，单位米                                │
├──────────────────────────────────────────────────────────┤
│ 阶段3：3D重建（3D Reconstruction）                        │
│   "真实世界长什么样？"                                      │
│                                                          │
│   多视角融合 → 场景点云 → 逐对象网格 → UV展开 → 纹理烘焙     │
│   输出：3D点云 + 相机位姿 + 带贴图的3D网格(.obj/.gltf)      │
├──────────────────────────────────────────────────────────┤
│ 阶段4：场景理解（Scene Understanding）                     │
│   "这是什么场景？地面在哪？重力方向？"                        │
│                                                          │
│   地面平面检测 → 重力对齐 → 物体分类 → 物理属性标注          │
│   输出：场景布局分析 {floor, walls, furniture, ...}        │
├──────────────────────────────────────────────────────────┤
│ 阶段5：导出（Export）                                     │
│   "目标引擎能读懂什么？"                                    │
│                                                          │
│   glTF / OBJ+MTL / Unity JSON / UE JSON / AE脚本 / PSD    │
│   输出：各目标平台可用的文件                                 │
└──────────────────────────────────────────────────────────┘
```

---

## 三、项目目录结构

```
ToThinkVision/
├── app/
│   ├── main.py                  # FastAPI入口，定义API路由
│   ├── config.py                # 全局配置（模型路径、设备、阈值）
│   ├── pipeline.py              # ★ 核心管线：串联所有模型
│   ├── schemas.py               # 数据结构定义（Pydantic模型）
│   ├── preprocessor.py          # 视频抽帧、图像预处理
│   │
│   ├── models/                  # AI模型模块
│   │   ├── sam3.py              # SAM 3：检测+分割+视频跟踪
│   │   ├── grounding_dino.py    # Grounding DINO：开放词汇目标检测
│   │   ├── omniparser.py        # OmniParser：UI元素检测
│   │   ├── depth_pro.py         # Depth Pro：单目深度估计
│   │   ├── mast3r.py            # MASt3R：3D点云+相机位姿重建
│   │   ├── gaussian_splatting.py # 3D高斯泼溅训练
│   │   ├── strongsort_wrapper.py # StrongSORT：多目标跟踪
│   │   └── mesh_reconstruction.py # ★ 逐对象3D网格重建（深度反投影→Poisson→UV→纹理）
│   │
│   ├── exporters/               # 导出模块
│   │   ├── gltf_exporter.py     # glTF 2.0（含UV+纹理+PBR材质）
│   │   ├── obj_exporter.py      # Wavefront OBJ + MTL
│   │   ├── game_exporter.py     # Unity JSON / UE JSON / 碰撞体
│   │   ├── ae_project_exporter.py # After Effects ExtendScript
│   │   ├── psd_exporter.py      # Photoshop PSD（静态/动态）
│   │   ├── ui_exporter.py       # Figma JSON / HTML-CSS
│   │   ├── video_exporter.py    # AE关键帧 / 轨迹CSV / PR标记
│   │   ├── embodied_exporter.py # 机器人动作JSON / 姿态CSV
│   │   ├── splat_exporter.py    # .splat二进制 / .ply高斯参数
│   │   ├── image_exporter.py    # 裁剪图 / mask / 深度可视化
│   │   ├── manifest.py          # 导出清单（README.txt）
│   │   └── base.py              # 导出器基类
│   │
│   └── utils/                   # 工具函数
│       ├── camera.py            # 相机内参/外参、坐标系变换、四元数
│       ├── geometry.py          # 碰撞检测、相对位置、z-index
│       ├── color.py             # 主色提取、RGB↔HEX转换
│       ├── pointcloud.py        # 深度反投影、法线估计、体素滤波
│       ├── texture_bake.py      # UV展开、多视角纹理烘焙
│       └── scene_understanding.py # 地面检测(RANSAC)、重力对齐
│
├── requirements.txt             # Python依赖
├── static/                      # 前端静态文件
└── tests/                       # 单元测试（51个）
```

---

## 四、需要学习的知识体系

### 第一层：基础（必须先学）

#### 1. 线性代数

**学什么：** 矩阵乘法、向量运算、坐标系变换、特征值分解、四元数

**在项目中的作用：**

- 相机位姿用 **4×4变换矩阵** 表示（3×3旋转 + 3×1平移）
- 3D点从一个坐标系变换到另一个坐标系（`app/utils/camera.py`）
- PCA法线估计用到 **特征值分解**（`_estimate_normals_pca`）
- 四元数表示旋转（避免欧拉角的万向节死锁）

**关键代码对应：**

```python
# app/utils/camera.py
def rt_matrix_to_position(R, t):
    """从旋转矩阵R和平移向量t计算相机在世界坐标系中的位置"""
    return -R.T @ t  # C = -R^T * t
```

**推荐学习：** 3Blue1Brown《线性代数的本质》（YouTube/B站，约3小时）

---

#### 2. 相机模型

**学什么：** 针孔相机模型、内参矩阵K、外参矩阵RT、像素坐标↔相机坐标↔世界坐标的三级变换

**在项目中的作用：**

- **内参矩阵 K**：包含焦距 (fx, fy) 和光心 (cx, cy)，决定"像素坐标如何映射到相机坐标"
  ```
  K = [fx   0  cx]
      [ 0  fy  cy]
      [ 0   0   1]
  ```
- **外参矩阵 RT**：4×4矩阵，描述相机在世界中的位置和朝向
- **深度反投影**：把2D像素坐标 + 深度值 → 3D点（`app/utils/pointcloud.py`）

**核心公式：**

```python
# 像素(u,v) + 深度d → 相机坐标系3D点(X,Y,Z)
X = (u - cx) * d / fx
Y = (v - cy) * d / fy
Z = d
```

**推荐学习：** 《Multiple View Geometry in Computer Vision》第2章（Hartley & Zisserman）

---

#### 3. Python + NumPy

**学什么：** 数组操作、广播机制、矩阵运算、索引切片

**项目里到处都是：** 所有3D数据都以 `(N, 3)` numpy数组形式流动。

---

### 第二层：感知（画面里有什么）

#### 4. 目标检测（Object Detection）

**学什么：** 边界框（bbox）、交并比（IoU）、非极大值抑制（NMS）、Transformer检测器

**项目中的模型：** `app/models/grounding_dino.py` — Grounding DINO

```
输入：RGB图像
输出：[{"bbox": [x, y, w, h], "label": "person", "confidence": 0.95}, ...]
```

**关键概念：** Open-vocabulary Detection — 不仅能检测训练时见过的类别，还能用自然语言描述来检测新物体（如"红色的椅子"）。

---

#### 5. 图像分割（Image Segmentation）

**学什么：** 语义分割 vs 实例分割 vs 可提示分割（Promptable Segmentation）

**项目中的模型：** `app/models/sam3.py` — SAM 3（Segment Anything Model）

```
输入：RGB图像 + 提示（bbox / 点 / 文本）
输出：逐像素二值mask（哪些像素属于目标物体）
```

**关键概念：** SAM不是自动检测物体，而是"你告诉我分割什么，我就帮你精确分割"。项目中先用Grounding DINO检测出bbox，再把bbox作为提示给SAM做精细分割。

---

#### 6. 多目标跟踪（Multi-Object Tracking）

**学什么：** 卡尔曼滤波（预测物体下一帧在哪）、匈牙利算法（匹配检测和轨迹）、ReID特征（外观相似度）

**项目中的模型：** `app/models/strongsort_wrapper.py` — StrongSORT

```
输入：第t帧的检测结果 + 第t-1帧的轨迹
输出：更新后的轨迹（track_id, 位置历史, 速度）
```

**输出数据结构：**

```python
TemporalInfo(
    appear_frame=15,        # 第15帧出现
    disappear_frame=120,    # 第120帧消失
    trajectory=[{x, y, t}, ...],  # 每帧中心位置
    depth_per_frame=[2.3, 2.5, ...], # 每帧深度
    velocity={"x": 0.5, "y": -0.1}   # 速度
)
```

---

### 第三层：深度与3D重建（真实世界长什么样）

#### 7. 单目深度估计（Monocular Depth Estimation）

**学什么：** 从单张RGB图估计每个像素的绝对深度（米）

**项目中的模型：** `app/models/depth_pro.py` — Depth Pro（Apple）

```
输入：单帧RGB图像
输出：深度图 (H, W)，单位米，近处亮/远处暗
```

**关键限制：** 单帧深度有"尺度模糊"——同一张图，房间可能是3米深也可能是30米深，模型只能估计相对深度。多视角重建可以解决这个问题。

---

#### 8. 多视角几何（Multi-View Geometry）★★★ 研究突破的核心区域

**学什么：** 对极几何、本质矩阵、三角测量、Bundle Adjustment、SfM（Structure from Motion）

**项目中的模型：** `app/models/mast3r.py` — MASt3R（NAVER）

```
输入：多帧图像（如每5帧取1帧）
输出：
  - 3D点云：场景的稀疏3D点 + 颜色
  - 相机位姿：每帧相机在世界中的位置和朝向
```

**关键概念：**

- **多视角约束：** 同一个3D点在不同帧的投影必须一致。利用这个约束，可以从2D图像反推出3D结构。
- **Bundle Adjustment（光束法平差）：** 同时优化3D点位置和所有相机位姿，使得所有3D点在各帧中的投影误差最小。
- **MASt3R的特殊之处：** 传统方法（COLMAP）需要提取SIFT特征→匹配→三角测量→BA，流程复杂。MASt3R用Transformer直接从图像对预测3D对应关系和置信度，端到端训练。

**这是项目最大的研究突破口：** 当前MASt3R在复杂实景视频（运动模糊、光照变化、遮挡多）中重建质量有限。

---

#### 9. 点云处理（Point Cloud Processing）

**学什么：** 体素滤波、法线估计、点云配准、离群点去除

**项目中的代码：** `app/utils/pointcloud.py`

```python
# 深度反投影：深度图 → 3D点云
points = backproject_depth(depth_map, K)  # (N, 3)

# 滤波：去掉太近/太远/重复的点
points, colors = filter_pointcloud(points, colors, voxel_size=0.02)

# 法线估计：每个点的表面朝向
normals = compute_normals(points[:50000], k=10)
```

**核心算法：**

- **体素下采样：** 把空间切成立方体网格，每个格子只保留一个点。大幅减少点数，同时保持几何形状。
- **PCA法线估计：** 对每个点的k个近邻做PCA，协方差矩阵最小特征值对应的特征向量就是法线方向（表面变化最小的方向）。

---

#### 10. 表面重建（Surface Reconstruction）

**学什么：** 从离散3D点生成连续三角网格的算法

**项目中的代码：** `app/models/mesh_reconstruction.py`

四种方法（质量从高到低）：

| 方法 | 原理 | 适用场景 | 需要 |
|------|------|---------|------|
| **Poisson** | 求解泊松方程，找零梯度曲面 | 封闭物体（球、杯子） | 需要法线 |
| **Marching Cubes** | 在体素网格上提取等值面 | 任意形状 | 需要体素化 |
| **Alpha Shape** | Delaunay三角剖分的子集 | 近似平面 | 无需额外输入 |
| **Convex Hull** | 凸包 | 最后兜底 | 无需额外输入 |

**项目中用的是 Poisson（Open3D）→ Marching Cubes → Alpha Shape → Convex Hull 的四级降级策略。**

---

#### 11. UV展开与纹理烘焙（UV Unwrapping & Texture Baking）

**学什么：** 把3D曲面"摊平"到2D平面；从多张照片采样颜色填入纹理图

**项目中的代码：** `app/utils/texture_bake.py`

```python
# UV展开：网格顶点 → 2D纹理坐标
uv_coords, uv_face_map = unwrap_uv(vertices, faces, method="box")

# 多视角纹理烘焙：把顶点投影回源视频帧，采样颜色
texture = bake_texture_multiview(
    vertices=vertices,
    faces=faces,
    uv_coords=uv_coords,
    frame_paths=frame_paths,
    camera_poses=camera_poses,
)
```

**关键概念：**

- **UV坐标：** 每个3D顶点对应纹理图上的一个2D点 `(u, v) ∈ [0, 1]²`。游戏引擎用这个坐标从纹理图上取颜色。
- **盒面投影：** 把模型沿法线方向投影到6个面（前/后/左/右/上/下），适合大多数物体。
- **纹理烘焙：** 对于每个3D顶点，找到所有能看到它的视频帧，按"视角正对程度 × 距离"加权混合颜色，避免单帧的颜色偏差。

---

### 第四层：场景理解（这是什么场景）

#### 12. 场景布局分析

**学什么：** RANSAC平面拟合、重力对齐、语义场景分类

**项目中的代码：** `app/utils/scene_understanding.py`

```python
# 从点云中检测地面平面
ground = detect_ground_plane(point_cloud_points)
# 输出：{normal: [0, 1, 0], point: [0, 0, 0], inlier_ratio: 0.4}

# 把场景旋转对齐到重力方向（Y轴朝上）
aligned_points, rotation_matrix = align_to_gravity(points, ground)

# 分析场景类型
layout = classify_scene_layout(objects, point_cloud_points)
# 输出：{"scene_type": "interior_room", "floor_detected": True, "wall_count": 4}
```

**算法核心：**

- **RANSAC平面拟合：** 随机取3个点拟合一个平面，统计有多少点在这个平面附近（inliers）。迭代多次，选inliers最多的平面。
- **重力对齐：** 计算地面法线到世界Y轴的旋转矩阵，把整个场景旋转过来，让"上"真正朝上。

---

### 第五层：导出（目标引擎能读懂什么）

#### 13. glTF 2.0 格式

**学什么：** 场景图、节点层次、Mesh/Accessor/BufferView的关系

**项目中的代码：** `app/exporters/gltf_exporter.py`

**核心结构：**

```
Scene (场景)
  └── Node (节点)
        └── Mesh (网格)
              └── Primitive (图元)
                    ├── attributes.POSITION → Accessor → BufferView → Buffer (顶点位置)
                    ├── attributes.NORMAL   → Accessor → BufferView → Buffer (法线)
                    ├── attributes.TEXCOORD_0 → Accessor → BufferView → Buffer (UV)
                    └── material → PBR材质 (baseColorTexture, metallic, roughness)
```

**为什么学这个：** glTF被称为"3D界的JPEG"，是Khronos Group（OpenGL/Vulkan的组织）推出的开放标准。Unity、Blender、Web浏览器（three.js）都原生支持。

---

#### 14. OBJ + MTL 格式

**学什么：** Wavefront OBJ的文件结构（v/vt/vn/f）、MTL材质定义

**项目中的代码：** `app/exporters/obj_exporter.py`

```obj
# 顶点
v 1.0 2.0 3.0
v 4.0 5.0 6.0
# UV坐标
vt 0.5 0.5
vt 0.0 0.0
# 法线
vn 0.0 1.0 0.0
# 面（顶点索引/UV索引/法线索引）
f 1/1/1 2/2/1 3/1/1
```

**最简单的3D交换格式**，所有3D软件都能读取。MTL文件定义材质属性（漫反射颜色、高光、透明度、纹理路径）。

---

#### 15. Unity/UE场景导入机制

**学什么：**

- Unity：MeshFilter（几何数据）+ MeshRenderer（材质渲染）+ MeshCollider（碰撞检测）
- UE：StaticMeshComponent + MaterialInterface

**项目中的代码：** `app/exporters/game_exporter.py`

当前导出的是JSON描述文件，Unity端需要配套C#脚本解析并创建GameObject。导出格式示例：

```json
{
  "name": "table",
  "tag": "Prop",
  "transform": {"position": {"x": 1.2, "y": 0.0, "z": 3.4}},
  "mesh_3d": {
    "has_mesh": true,
    "vertices": 5230,
    "mesh_file": "meshes/obj_0001.obj",
    "texture_path": "meshes/obj_0001_texture.png"
  },
  "components": [
    {"type": "MeshFilter"},
    {"type": "MeshRenderer", "material": {"texture": "obj_0001_texture.png"}},
    {"type": "MeshCollider", "convex": true}
  ]
}
```

---

#### 16. After Effects ExtendScript

**学什么：** AE的JavaScript API、合成（Composition）、图层（Layer）、关键帧（Keyframe）

**项目中的代码：** `app/exporters/ae_project_exporter.py`

输出一个 `.jsx` 脚本 + 数据JSON。在AE中运行脚本会自动创建：

- 原始视频作为半透明参考图层
- 每个检测到的物体作为图片图层，带透明背景（mask）
- 位置关键帧（来自StrongSORT跟踪轨迹）
- 透明度关键帧（物体出现/消失的帧）
- 3D摄像机（来自MASt3R相机位姿）

---

### 第六层：进阶研究方向

#### 17. 3D Gaussian Splatting（3DGS）

**项目中的代码：** `app/models/gaussian_splatting.py`

**核心思想：** 场景用数十万个3D高斯函数表示。每个高斯有：位置(3)、旋转(4)、缩放(3)、不透明度(1)、球谐颜色系数(C)。渲染时用"可微分光栅化"把3D高斯投影到2D屏幕。

**与NeRF的对比：**

| | NeRF | 3DGS |
|---|------|------|
| 表示方式 | 隐式（神经网络） | 显式（高斯点集） |
| 训练速度 | 慢（小时级） | 快（分钟级） |
| 渲染速度 | 慢（需要推理网络） | 实时（60+ FPS） |
| 质量 | 高 | 中高 |
| 可编辑性 | 差（黑盒） | 较好（可单独编辑高斯） |

**项目中的限制：** 当前训练循环是手写的简化版。要达到商用水准，建议集成COLMAP做相机位姿预处理 + nerfstudio做训练。

---

#### 18. NeRF（Neural Radiance Fields）

**核心思想：** 用一个MLP学习场景的体积密度和颜色。输入是3D点+视角方向，输出是该点的密度和颜色。

**与项目的关系：** 可以从NeRF密度场中提取高质量网格（marching cubes），比当前深度反投影方法精细得多。

---

#### 19. SLAM / VIO（同步定位与建图）

**核心思想：** 实时估计相机位姿的同时构建环境的3D地图。

**与项目的关系：** 当前用MASt3R离线重建（处理完整个视频才出结果）。SLAM可以实时/增量式重建，更精确的相机位姿。

**相关工具：** ORB-SLAM3、COLMAP、OpenVSLAM

---

#### 20. 多模态大模型（VLM）

**核心思想：** GPT-4V、Qwen-VL等模型不仅能理解2D图像，还能进行3D推理。

**研究突破口：** 用VLM做物体识别和语义分割，替代Grounding DINO；用VLM理解场景布局（"这是厨房，有桌子和椅子"）替代手工规则。

---

## 五、知识点之间的关系

```
线性代数 ─────────────────────────────────────────┐
    │                                              │
    ▼                                              ▼
相机模型 ──→ 内参K + 外参RT ──→ 三级坐标变换         │
    │                          │                    │
    ▼                          ▼                    │
深度反投影 ──→ 相机坐标3D点     MASt3R ──→ 世界坐标   │
    │                          │                    │
    └───────→ 合并点云 ←───────┘                    │
                │                                   │
                ▼                                   │
        点云处理（滤波/法线）                         │
                │                                   │
                ▼                                   │
        表面重建（Poisson/MC）                        │
                │                                   │
                ▼                                   │
          网格(Mesh)                                 │
            /      \                                 │
           /        \                                │
       UV展开    多视角纹理烘焙 ←── 相机位姿 ─────────┘
           \        /
            \      /
          纹理贴图(PNG)
                │
                ▼
        ┌─── glTF / OBJ / Unity / AE / PSD ───┐
        │                                       │
        ▼                                       ▼
   Blender/Unity                            After Effects
   游戏引擎                                  影视后期
```

---

## 六、推荐学习路径

| 阶段 | 学习内容 | 预计时间 | 验证方式 |
|------|---------|---------|---------|
| **1** | 线性代数基础 + 相机模型 | 2周 | 能手推 `backproject_depth` 公式，解释K矩阵每个参数的含义 |
| **2** | 读通当前项目代码 | 1周 | 能说出 `pipeline.py` 每一段在做什么，画出数据流图 |
| **3** | 目标检测 + SAM + 跟踪原理 | 2周 | 能跑通单张图片的检测+分割，理解IoU和NMS |
| **4** | 深度估计 + 点云处理 | 2周 | 能理解深度图→点云→网格的完整转换链 |
| **5** | 多视角几何 + MASt3R原理 | 3周 | 能解释Bundle Adjustment优化什么，SfM流程的每一步 |
| **6** | UV展开 + 纹理烘焙 | 1周 | 能手工给一个立方体贴UV，理解barycentric interpolation |
| **7** | glTF格式 + 导出流程 | 1周 | 能手写一个最简单的glTF文件并在浏览器中打开 |
| **8** | 3DGS / NeRF（研究突破方向） | 4周+ | 能比较不同重建方法的优劣，指出项目改进方向 |

### 每个阶段的实践任务

**阶段1-2实践：**
- 用 `python -c "import numpy as np; ..."` 练习矩阵乘法
- 运行 `app/utils/camera.py` 中的函数，看内参矩阵长什么样
- 读 `pipeline.py`，在纸上画出数据从输入到输出的流向

**阶段3实践：**
- 下载一张图片，运行 `detector.detect(img)` 看检测结果
- 运行 `sam3.predict(img, boxes=...)` 看分割mask
- 理解 `per_frame_objects` 字典如何在多帧间累积

**阶段4实践：**
- 运行 `depth_model.estimate(img)` 看深度图
- 运行 `backproject_depth(depth_map, K)` 看3D点云
- 理解 `_filter_outliers` 去掉了哪些点

**阶段5实践：**
- 运行 `reconstructor.reconstruct(frame_dir)` 看MASt3R重建效果
- 对比有/无相机位姿时的网格质量差异
- 理解 `reconstruct_object_meshes` 的6步流程

**阶段6-7实践：**
- 运行 `unwrap_uv(vertices, faces)` 看UV展开结果
- 运行 `bake_texture_multiview(...)` 看纹理烘焙效果
- 导出glTF文件，在 https://gltf-viewer.donmccurdy.com/ 中打开

---

## 七、最容易出研究突破的方向

### 1. 多视角重建质量改进

**现状问题：** MASt3R在复杂实景视频（运动模糊、光照变化、遮挡多）中重建稀疏且不完整。

**可能的方向：**
- 结合NeRF/3DGS的隐式表示，从密度场提取网格（比深度反投影精细）
- 集成COLMAP做更精确的相机位姿估计
- 多尺度融合：MASt3R做全局结构，深度图做局部细节

### 2. 3DGS训练管线升级

**现状问题：** 当前训练循环是手写简化版，缺少COLMAP相机位姿预处理。

**改进方案：** 集成COLMAP（相机位姿）+ nerfstudio（训练框架），替代手写循环。

### 3. 语义辅助重建

**现状问题：** 地面/墙面检测只用几何（RANSAC），在复杂场景容易失败。

**改进方案：** 用VLM（如Qwen-VL、GPT-4V）先理解场景语义（"这是厨房，地面在画面下半部分"），再用几何方法精确检测。

### 4. 实时性/增量式重建

**现状问题：** 当前是全离线处理——处理完整个视频才出结果。

**改进方向：** 流式处理——每来一帧就更新场景，支持实时监控和交互式编辑。

### 5. Unity/UE编辑器插件

**现状问题：** 当前只导出JSON描述文件，需要手动编写C#脚本导入。

**改进方向：** 开发Unity Editor插件，一键导入ToThinkVision的JSON + OBJ + 纹理，自动创建场景。

---

## 八、常用资源与参考链接

### 基础知识

| 资源 | 内容 | 链接 |
|------|------|------|
| 3Blue1Brown | 线性代数的本质（视频） | YouTube/B站搜索 |
| Multiple View Geometry | 多视角几何圣经（书） | Hartley & Zisserman |
| Stanford CS231n | 计算机视觉课程 | cs231n.stanford.edu |
| NumPy官方文档 | 数组操作手册 | numpy.org/doc |

### 模型与工具

| 工具 | 用途 | 链接 |
|------|------|------|
| SAM 3 | 分割 | github.com/facebookresearch/sam3 |
| Grounding DINO | 开放词汇检测 | github.com/IDEA-Research/GroundingDINO |
| Depth Pro | 深度估计 | github.com/apple/ml-depth-pro |
| MASt3R | 3D重建 | github.com/naver/mast3r |
| COLMAP | SfM/BA | colmap.github.io |
| nerfstudio | 3DGS/NeRF训练 | docs.nerf.studio |
| Open3D | 3D处理库 | open3d.org |
| three.js | Web 3D渲染 | threejs.org |
| glTF Viewer | 在线查看glTF | gltf-viewer.donmccurdy.com |

### glTF/OBJ格式参考

| 资源 | 链接 |
|------|------|
| glTF 2.0 Specification | github.com/KhronosGroup/glTF/blob/main/specification/2.0 |
| OBJ Format Reference | paulbourke.net/dataformats/obj |
| MTL Format Reference | paulbourke.net/dataformats/mtl |

### AE ExtendScript

| 资源 | 链接 |
|------|------|
| AE Scripting Guide | helpx.adobe.com/after-effects/using/scripting.html |
| AE ExtendScript Toolkit | github.com/Adobe-CEP |
