# 机器学习课程大作业代码 - BSPM模型复现

## 项目概述

本项目是**机器学习课程期末大作业**的代码实现部分。我们选择了**BSPM**(Blurring-Sharpening Process Model)模型进行复现，该模型是一种创新的协同过滤推荐算法，通过常微分方程实现推荐系统的扰动-恢复架构。

## BSPM模型简介

BSPM是一种基于常微分方程的协同过滤推荐算法，采用**扰动-恢复架构**：

- **模糊过程**：对用户-物品交互矩阵进行平滑处理，模拟信息在图上的聚合
- **锐化过程**：逆转平滑效果，提取用户个性化特征，增强长尾物品推荐能力

### 核心优势

- **无需训练** - 不涉及反向传播，推理速度极快
- **连续时间建模** - 使用常微分方程（欧拉法/RK4）进行精确求解
- **高准确率** - 在多个数据集上达到或超过传统图卷积模型的效果
- **高可解释性** - 基于数学公式，具有明确的物理意义

### 原论文信息

- 论文标题：Blurring-Sharpening Process Models for Collaborative Filtering
- 原论文地址：[Choi J, Hong S, Park N, et al. (2023)](https://dl.acm.org/doi/10.1145/3539618.3591645)

## 技术框架

本项目基于 **ReChorus** 框架进行开发，该框架专门用于推荐系统算法的复现和对比。

- ReChorus框架GitHub链接：[https://github.com/THUwangcy/ReChorus](https://github.com/THUwangcy/ReChorus)
- 我们在框架基础上实现了BSPM算法，并进行了以下优化：
  - 使用PyTorch稀疏矩阵减少显存占用
  - 设计缓存机制提升重复推理速度
  - 重新组织矩阵运算降低内存压力

## 运行方式

### 基本命令

```bash
python src/main.py --model_name BSPM --workers 4 --train 0 --dataset Grocery_and_Gourmet_Food
```

### 支持的数据集

- `Grocery_and_Gourmet_Food` - Amazon 食品类商品数据集
- `MovieLens_1M` - 电影评分数据集

