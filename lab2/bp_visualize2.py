from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
from pandas import DataFrame
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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

# 代码的手动实现
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # 初始化权重
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)

        # 初始化偏置
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))

    def sigmoid(self, x):  # sigmoid 计算方式
        return 1 / (1 + np.exp(-x))
        return

    def sigmoid_derivative(self, x):  # sigmoid 导数计算方式
        return x * (1 - x)
        return

    def forward(self, X):
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)  # 隐藏层输出
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.output = self.sigmoid(self.output_layer_input)  # 输出层输出
        return self.output
        
    def backward(self, X, y, output, learning_rate):
        # 反向传播更新权重和偏置
        output_error = y - output  # 输出层误差
        output_delta = output_error * self.sigmoid_derivative(output)  # 输出层梯度

        hidden_error = output_delta.dot(self.weights_hidden_output.T)  # 隐藏层误差
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_layer_output)  # 隐藏层梯度

        # 更新隐藏层到输出层的权重和偏置
        self.weights_hidden_output += self.hidden_layer_output.T.dot(output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate

        # 更新输入层到隐藏层的权重和偏置
        self.weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate


    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward(X)
            loss = np.mean(0.5 * (y - output) ** 2)
            self.backward(X, y, output, learning_rate)
            if epoch % 100 == 0:
                print(f"Epoch {epoch + 1}, Loss: {loss}")

    def predict(self, X):
        return np.round(self.forward(X))


# 将标签转换为独热编码
def one_hot_encode(labels):
    num_classes = len(np.unique(labels))
    one_hot_labels = np.zeros((len(labels), num_classes))
    for i, label in enumerate(labels):
        one_hot_labels[i][label] = 1
    return one_hot_labels


# 构建神经网络
input_size = X_train.shape[1]
hidden_size = 10
output_size = len(np.unique(Y_train))  # 根据训练集标签确定输出层大小
nn = NeuralNetwork(input_size, hidden_size, output_size)

# 将标签转换为独热编码
Y_train_encoded = one_hot_encode(Y_train)

# 训练神经网络
print("training.......")
nn.train(standard_train, Y_train_encoded, epochs=2000, learning_rate=0.01)
# 预测测试集
predictions = nn.predict(standard_test)

# 计算准确率
accuracy = accuracy_score(Y_test, np.argmax(predictions, axis=1))

# 查看模型结果
print("测试集合的 y 值：", list(Y_test))
print("神经网络预测的的 y 值：", list(np.argmax(predictions, axis=1)))
print("预测的准确率为：", accuracy)

# 可视化：PCA 降维后的数据分布
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

# 1. 绘制训练过程中的损失曲线
def plot_loss_curve(nn, X, y, epochs, learning_rate):
    losses = []
    for epoch in range(epochs):
        output = nn.forward(X)
        loss = np.mean(0.5 * (y - output) ** 2)
        nn.backward(X, y, output, learning_rate)
        losses.append(loss)
        if epoch % 100 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss}")
    plt.figure(figsize=(8, 6))
    plt.plot(range(epochs), losses, label="Loss Curve")
    plt.title("Training Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()

# 绘制训练损失曲线
plot_loss_curve(nn, standard_train, Y_train_encoded, epochs=2000, learning_rate=0.01)

# 2. 预测结果的真实值 vs 预测值
def plot_predictions(y_test, predictions):
    plt.figure(figsize=(10, 6))
    plt.plot(list(y_test), 'o-', label="True Values", alpha=0.7)
    plt.plot(np.argmax(predictions, axis=1), 'x-', label="Predicted Values", alpha=0.7)
    plt.title("True vs Predicted Values")
    plt.xlabel("Sample Index")
    plt.ylabel("Target Value")
    plt.legend()
    plt.grid()
    plt.show()

# 调用函数绘制预测结果对比
plot_predictions(Y_test, predictions)

# 3. 绘制混淆矩阵
def plot_confusion_matrix(y_test, predictions, class_names):
    cm = confusion_matrix(y_test, np.argmax(predictions, axis=1))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.show()

# 绘制混淆矩阵
plot_confusion_matrix(Y_test, predictions, iris.target_names)

# 4. 可视化网络权重分布
def plot_weights(nn):
    plt.figure(figsize=(10, 6))
    plt.imshow(nn.weights_input_hidden, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title("Input to Hidden Layer Weights")
    plt.xlabel("Hidden Neurons")
    plt.ylabel("Input Features")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.imshow(nn.weights_hidden_output, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title("Hidden to Output Layer Weights")
    plt.xlabel("Output Neurons")
    plt.ylabel("Hidden Neurons")
    plt.show()

# 调用函数可视化权重
plot_weights(nn)