# AMASS — MoCap Retargeting Pipeline

将 CMU 动作捕捉数据 (c3d) 通过 MoSh++ 转换为 AMASS 标准格式 (SMPL-X/SMPL-H)，并支持异构动捕数据向 SMPL 骨架的重定向。

## 项目结构

```
amass/
├── input/                    # 原始 c3d 动捕文件
│   ├── 87_03.c3d
│   └── 87_04.c3d
├── support_data/             # SMPL-X 模型文件 & 先验数据
│   └── smplx/
│       ├── male/model.pkl
│       ├── female/model.pkl
│       ├── neutral/model.pkl
│       └── pose_hand_prior.npz
├── work/                     # 中间结果 & 输出
│   ├── CMU/c3d/subjects/     # CMU 数据集缓存
│   ├── mocap/                # 异构动捕参考数据 (待重定向)
│   │   └── 0416.npz
│   └── mosh_results/         # MoSh++ Stage I/II 输出
├── tests/                    # 测试和调试脚本
│   ├── bench.py              # forward pass 性能基准
│   ├── test_forward.py       # 前向传播测试
│   ├── test_lbs.py           # LBS 蒙皮测试
│   ├── test_step.py          # 单步优化测试
│   ├── test_stageii.py       # Stage II 测试
│   ├── debug_jac.py          # Jacobian 调试
│   └── debug_jac2.py         # Jacobian 调试 2
├── run_mosh.py               # 标准 MoSh++ 双阶段流程
├── convert_cmu.py            # CMU c3d → AMASS npz (完整流程)
├── convert_cmu_direct.py     # 同上 (scipy 优化器, L-BFGS-B)
├── convert_minimal.py        # 最小化测试流程 (极小迭代)
├── fast_convert.py           # 快速转换 (跳过 chumpy, ~87x 加速)
└── .gitignore
```

## 环境准备

### 1. 安装依赖

```bash
pip install numpy scipy scikit-learn chumpy omegaconf
```

### 2. 安装 MoSh++

```bash
git clone https://github.com/nghorbani/moshpp.git /home/user/retargeting/moshpp
# 或放置于其他路径，需同步修改脚本中的 sys.path
```

### 3. 下载 SMPL-X 模型

SMPL-X 模型需从 [MPI-IS](https://smpl-x.is.tue.mpg.de/) 注册下载。下载后将以下文件放入对应目录：

```text
support_data/smplx/
├── male/model.pkl        # 男性模型 (10475 顶点, 55 关节)
├── female/model.pkl      # 女性模型
├── neutral/model.pkl     # 中性模型
└── pose_hand_prior.npz   # 手部姿态先验 (MANO)
```

> 注意：模型文件受许可保护，未纳入 Git。`.gitignore` 已排除 `support_data/smplx/*/model.pkl`。

## 用法

### 1. 标准 MoSh++ 双阶段转换

```bash
python run_mosh.py
```

对 `work/CMU/c3d/subjects/87/*.c3d` 中的 c3d 文件依次执行：
- **Stage I**: 估算体型参数 (betas) —— 基于少帧 marker 数据
- **Stage II**: 逐帧优化姿态参数 (axis-angle) + 位移 (trans)
- **输出**: `<basename>_poses.npz` (AMASS 格式)

### 2. 快速转换 (绕过 chumpy)

```bash
python fast_convert.py [c3d_file1 c3d_file2 ...]
```

- 仅计算 marker 投影所需的 ~120 个顶点 (vs 10475 全量)，约 **87x 加速**
- 使用 scipy L-BFGS-B + 纯 numpy forward pass
- 依赖已完成 Stage I 的 `male_stagei.pkl`

### 3. 配置选项

在 `convert_cmu.py` 中的 `job` 字典可调整：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `surface_model.type` | 模型类型 | `smplx` |
| `surface_model.gender` | 性别 | `male` |
| `moshpp.optimize_betas` | 优化体型 | `True` |
| `moshpp.optimize_fingers` | 优化手指 | `False` |
| `moshpp.optimize_face` | 优化面部 | `False` |
| `opt_settings.maxiter` | 最大迭代次数 | `30` |

## 输出格式 (AMASS npz)

| 字段 | 形状 | 说明 |
|------|------|------|
| `gender` | str | 性别 |
| `surface_model_type` | str | 模型类型 (smplx) |
| `trans` | [F, 3] | 全局位移 |
| `poses` | [F, J×3] | 全部关节轴角旋转 (SMPL-X: 165维) |
| `betas` | [B] | 体型参数 |
| `root_orient` | [F, 3] | 根关节旋转 |
| `pose_body` | [F, 63] | 身体关节 (21 × 3) |
| `pose_hand` | [F, 90] | 手部关节 |
| `mocap_frame_rate` | float | 帧率 |
| `mocap_time_length` | float | 时长 (秒) |

## 异构动捕重定向 (retargeting)

`work/mocap/0416.npz` 为外部动捕数据，与 SMPL 存在以下差异：

| 维度 | 外部动捕 | SMPL-H |
|------|---------|--------|
| 关节数 | 33 | 52 |
| 骨架拓扑 | 未知层级 | kintree_table 定义 |
| 坐标系 | 待定 | SMPL 标准 |
| 数据格式 | `raw_joint_data` [F, 99] | AMASS npz |

重定向流程将在 `retargeting` 分支上实现。

## License

MIT
