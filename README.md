# Eigenfaces for Recognition 

本项目是针对经典计算机视觉论文 **Turk & Pentland (1991) "Eigenfaces for Recognition"** 的完整复现与扩展实验系统。通过主成分分析（PCA）构建特征脸空间，并在 **ORL (AT&T)** 和 **Yale** 两个经典人脸数据集上进行了多维度的性能评估。

本系统不仅复现了论文的核心结论，还进一步探索了超参数（特征向量个数 $K$）、训练样本量、人脸/非人脸分类以及光照变化对识别率的影响。

---

## 📂 项目文件结构

```text
eigenfaces-recognition/
  ├── README.md                    # 项目说明文档（本文件）
  ├── eigenfaces_reproduce.py       # 主程序代码（包含完整的PCA、训练、测试及绘图逻辑）
  ├── results_orl/                 # ORL 数据集实验生成的图表及可视化结果
  └── results_yale/                # Yale 数据集实验生成的图表及可视化结果
```

## 📊 评测数据集配置

```text
- ORL/AT&T Face Database：40人×10张，用于完整算法性能分析
  下载：https://cam-orl.co.uk/facedatabase.html
- Yale Face Database：15人×11种条件，用于Figure 9光照实验复现
  下载：http://vision.ucsd.edu/content/yale-face-database
```
## 📈 核心实验结果
### ORL数据集（7组实验）
| 实验 | 内容 | 关键结果 |
|------|------|---------|
| Exp0 | 平均脸+特征脸+重建 | 基础准确率88.50% |
| Exp1 | 准确率 vs K值 | 最优K=32 |
| Exp2 | 准确率 vs 训练量 | 每人7张→95.83% |
| Exp3 | 多曲线对比 | 训练量↑最优K↓ |
| Exp4 | 重建误差MSE | K越大误差越小 |
| Exp5 | 人脸vs非人脸 | 距离比值9.9x |
| Exp7 | 数据库规模 | 人数↑准确率↓ |
| Exp8 | 交叉验证 | 结果稳定可靠 |

### Yale数据集
光照变化对识别影响显著，验证论文核心结论。
