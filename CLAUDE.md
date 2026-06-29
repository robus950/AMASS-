# AMASS — MoCap Retargeting Pipeline

将 CMU 动作捕捉数据 (c3d) 通过 MoSh++ 转换为 AMASS 标准格式 (SMPL-X/SMPL-H)，并支持异构动捕数据向 SMPL 骨架的重定向。

## 技术栈

- **Python**: 3.11+ (amass conda 环境)
- **MoSh++**: CMU mocap → AMASS/SMPL 转换
- **SMPL-X/SMPL-H**: 人体参数化模型
- **PyTorch**: 数值优化 / LBS 蒙皮
- **NumPy**: 数值计算

## 项目结构

```
amass/
├── input/                    # 原始 c3d 动捕文件 (.c3d)
├── support_data/smplx/       # SMPL-X 模型文件 (.pkl 三个性别)
├── work/                     # 中间结果和输出
│   ├── CMU/c3d/subjects/     # CMU 数据集缓存
│   ├── mocap/                # 异构动捕参考数据 (.npz)
│   └── mosh_results/         # MoSh++ Stage I/II 输出
├── retargeting/              # 动捕重定向核心模块
│   ├── build_amass.py        # AMASS 格式构建
│   ├── retarget.py           # 重定向主逻辑
│   ├── recover.py            # SMPL 参数恢复
│   ├── source_fk.py          # 源骨架前向运动学
│   ├── skeleton_def.py       # 骨架定义 (关节映射)
│   ├── coord_convert.py      # 坐标系转换
│   ├── parse_npz.py          # NPZ 文件解析
│   ├── fbx_pre_rotations.py  # FBX 预旋转处理
│   ├── visualize_verify.py   # 可视化验证
│   ├── test_retarget.py      # 重定向测试
│   └── XBR/                  # XBR 机器人模型 (FBX/URDF/mesh)
├── tests/                    # 测试和调试脚本
├── convert_cmu*.py           # CMU c3d 转换脚本
├── fast_convert.py           # 快速批量转换
└── run_mosh.py               # MoSh++ 运行入口
```

## 关键命令

```bash
# 激活环境
conda activate amass

# 转换 CMU c3d 数据
python convert_cmu_direct.py

# 运行重定向
python -m retargeting.retarget

# 运行测试
python tests/test_forward.py
```

## Conda 环境

环境名: `amass`
Python: 3.11
关键依赖: pytorch, numpy, scipy, smplx, psbody-mesh

## Git 仓库

远程: `git@github.com:robus950/AMASS-.git`
分支: `main`
