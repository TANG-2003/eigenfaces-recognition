# Eigenfaces for Recognition (论文复现实验)

本项目是针对经典计算机视觉论文 **Turk & Pentland (1991) "Eigenfaces for Recognition"** 的完整复现与扩展实验系统。通过主成分分析（PCA）构建特征脸空间，并在 **ORL (AT&T)** 和 **Yale** 两个经典人脸数据集上进行了多维度的性能评估。

本系统不仅复现了论文的核心结论，还进一步探索了超参数（特征向量个数）、训练样本量、人脸/非人脸分类以及光照变化对识别率的影响。
## 📂 项目文件结构

```text
eigenfaces-recognition/
  ├── README.md                    # 项目说明文档（本文件）
  ├── eigenfaces_reproduce.py       # 主程序代码（包含完整的PCA、训练、测试及绘图逻辑）
  ├── results_orl/                 # ORL 数据集实验生成的图表及可视化结果
  └── results_yale/                # Yale 数据集实验生成的图表及可视化结果

### 🖼️ 核心实验图表直观展示

ORL 数据集：特征数量 $K$ 对识别准确率的影响 (Exp1)
![Accuracy vs K-value](results_orl/pca_accuracy.png)
*注：由上图可见，随着特征脸数量 $K$ 的增加，系统识别率在初始阶段迅速跃升，并在 $K=32$ 附近达到饱和平台期，完美验证了 PCA 降维在信息保留上的高效性。*

Yale 数据集：极端光照条件下的识别表现 (Figure 9 复现)
![Yale Lighting Experiment](results_yale/figure9_produced_final.png)
*注：上图展示了系统在应对单侧强光（如 `leftlight`）时的局部局限性。由于经典 Eigenfaces 过于依赖像素级的线性投影，极端阴影会剧烈改变面部特征分布，这有力支撑了原论文关于光照鲁棒性挑战的论述。*
