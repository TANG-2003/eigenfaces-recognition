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
