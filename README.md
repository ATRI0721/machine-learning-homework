# BSPM (Blurring-Sharpening Process Model)

## 模型简介

BSPM 是一种基于常微分方程的协同过滤推荐算法，采用**扰动-恢复架构**：

- **模糊过程**：对用户-物品交互矩阵进行平滑处理，模拟信息在图上的聚合
- **锐化过程**：逆转平滑效果，提取用户个性化特征，增强长尾物品推荐能力

### 核心优势

- **无需训练** - 不涉及反向传播，推理速度极快（比 LightGCN 快 10 倍以上）
- **连续时间建模** - 使用常微分方程（欧拉法/RK4）进行精确求解
- **高准确率** - 在多个数据集上达到或超过传统图卷积模型的效果
- **高可解释性** - 基于数学公式，具有明确的物理意义

## 运行方式

### 基本命令

```bash
python src/main.py --model_name BSPM --workers 4 --train 0 --dataset Grocery_and_Gourmet_Food
```

### 支持的数据集

- `Grocery_and_Gourmet_Food` - Amazon 食品类商品数据集
- `MovieLens_1M` - 电影评分数据集

