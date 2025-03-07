# 待办事项 (TODO List)

## 优化与改进

- [ ] 加入其他优化算法：
  - [X] 遗传算法（GA）：模拟自然选择和遗传机制，适用于复杂优化问题。
  - [X] 模拟退火算法（SA）：模拟金属退火过程，以一定概率接受较差解，有助于跳出局部最优。
  - [X] 差分进化算法（DE）：基于种群的优化算法，通过个体差异进行搜索，具有较强的全局搜索能力。
  - [X] Try XGBoost / LightGBM (better than Random Forest for small data).
  - [X] Improve Features (PCA, feature interactions).
  - [X] Use ADASYN instead of SMOTE (avoids noisy synthetic data).
  - [ ] Use Bayesian Optimization for Hyperparameter Tuning .
  - [ ] ExtraTreesEntr
  - [X] Try an Ensemble (Voting Classifier) for better generalization.

## 已完成

- [X] 三比值法特征工程处理
- [X] 比值差分构造特征工程处理
- [X] 分类器树
- [X] 分类器SVM
- [ ] 添加SHAP方法，用于模型解释性分析 Why 贡献高的原因？算法角度
- [ ] 数据集固定划分
- [X] 使用SMOTE进行数据平衡，解决类别不平衡问题
- [X] 尝试使用SMOTE的变种进行数据平衡：
  - [X] Borderline-SMOTE：关注边界样本，提高分类器对边界样本的识别能力。
  - [ ] ADASYN：根据样本的难易程度生成不同数量的合成样本，更关注难以学习的样本。
  - [X] KMeansSMOTE：使用K-means聚类算法生成更具代表性的合成样本。
- [X] 加入粒子群优化（PSO）算法，用于超参数调优或特征选择
- [ ] 时间序列 数学定义 特征+++
- [ ] 指标：一致性F1 Acc precision R PR 对比
- [ ] 参数优化
- [ ]

## 参考

[AutoGluon](https://auto.gluon.ai/stable/tutorials/tabular/tabular-quick-start.html)
