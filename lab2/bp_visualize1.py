import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import datasets
from pandas import DataFrame

# 加载数据集
iris = datasets.load_iris()
df = DataFrame(iris.data, columns=iris.feature_names)
df["target"] = list(iris.target)
X = df.iloc[:, 0:4]
Y = df.iloc[:, 4]

# 划分数据集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# 数据标准化
sc = StandardScaler()
sc.fit(X)
standard_train = sc.transform(X_train)
standard_test = sc.transform(X_test)

# 构建 MLP 模型
mlp = MLPClassifier(hidden_layer_sizes=(10,10), activation='logistic', max_iter=2000, random_state=0)
mlp.fit(standard_train, Y_train)
result = mlp.predict(standard_test)

# 准确率
accuracy = mlp.score(standard_test, Y_test)
print(f"模型的准确率: {accuracy}")

# 可视化 1：PCA 降维后的数据分布
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
colors = ['red', 'blue', 'green']
plt.figure(figsize=(10, 6))
for i, target_name in enumerate(iris.target_names):
    plt.scatter(X_pca[Y == i, 0], X_pca[Y == i, 1], label=target_name, color=colors[i], alpha=0.6)
plt.title("PCA Visualization of Iris Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.show()

# 可视化 2：模型的收敛曲线
loss_curve = mlp.loss_curve_
plt.figure(figsize=(8, 6))
plt.plot(loss_curve, label="Loss Curve", color="orange")
plt.title("Model Training Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# 可视化 3：权重矩阵和偏置可视化
weights, biases = mlp.coefs_, mlp.intercepts_

# 绘制隐藏层权重的热力图
plt.figure(figsize=(10, 6))
plt.imshow(weights[0], aspect='auto', cmap='viridis')
plt.colorbar()
plt.title("Weights Heatmap (Input to Hidden Layer)")
plt.xlabel("Hidden Neurons")
plt.ylabel("Input Features")
plt.show()

# 绘制输出层权重的热力图
plt.figure(figsize=(10, 6))
plt.imshow(weights[1], aspect='auto', cmap='viridis')
plt.colorbar()
plt.title("Weights Heatmap (Hidden to Output Layer)")
plt.xlabel("Output Neurons")
plt.ylabel("Hidden Neurons")
plt.show()
