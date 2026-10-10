import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, silhouette_score
import time

# 1. 手写实现 K-Means 算法
class SimpleKMeans:
    def __init__(self, k=3, max_iters=100, tol=1e-4, random_state=42):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels = None

    def fit(self, X):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape

        # 1. 初始化：随机选择 k 个样本作为初始质心
        # (这也是最朴素的初始化，后续可以优化为 K-Means++)
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices]

        for i in range(self.max_iters):
            # 2. 分配簇：计算每个样本到所有质心的距离
            # 使用广播机制计算欧氏距离: (samples, 1, features) - (k, features)
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)

            # 获取最近质心的索引
            self.labels = np.argmin(distances, axis=1)

            # 3. 更新质心：计算每个簇的平均值
            new_centroids = np.array([X[self.labels == j].mean(axis=0) for j in range(self.k)])

            # 4. 检查收敛：如果质心变化很小，则停止
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                print(f"K-Means 在第 {i + 1} 轮迭代收敛。")
                break

            self.centroids = new_centroids

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

# 2. 数据准备与辅助函数

def load_and_filter_mnist():
    print("正在加载 MNIST 数据集 (可能需要一点时间)...")
    # 为了演示速度，我们只取前20000个样本进行筛选，实际使用可以取更多
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data, mnist.target

    # 筛选标签为 '1', '4', '8' 的数据
    mask = np.isin(y, ['1', '4', '8'])
    X_filtered = X[mask]
    y_filtered = y[mask].astype(int)

    # 归一化像素值到 0-1 之间
    X_filtered = X_filtered / 255.0

    print(f"筛选后数据形状: {X_filtered.shape}")
    print(f"包含标签: {np.unique(y_filtered)}")
    return train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=42)


def evaluate_clustering(y_true, y_pred, algorithm_name="K-Means"):
    """
    由于 K-Means 是无监督的，预测的 label (0,1,2) 和真实的 label (1,4,8) 不一定对应。
    我们需要找到一种映射关系来计算准确率（纯度）。
    """
    from scipy.stats import mode

    # 创建一个映射后的预测标签数组
    labels = np.zeros_like(y_pred)

    # 对于每个聚类簇，找到该簇中出现次数最多的真实标签
    for i in range(3):
        mask = (y_pred == i)
        if np.sum(mask) > 0:
            # 找出该簇中众数对应的真实标签
            most_frequent = mode(y_true[mask], keepdims=True)[0][0]
            labels[mask] = most_frequent

    acc = accuracy_score(y_true, labels)
    sil = silhouette_score(X_test, y_pred) if len(X_test) < 10000 else "Skipped (too slow)"

    print(f"--- {algorithm_name} 结果 ---")
    print(f"聚类纯度/准确率 (Purity): {acc:.4f}")
    if isinstance(sil, float):
        print(f"轮廓系数 (Silhouette): {sil:.4f}")
    print("混淆矩阵:\n", confusion_matrix(y_true, labels))
    return labels  # 返回修正后的标签以便绘图


def plot_results(X_vis, y_pred, title):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_pred, cmap='viridis', alpha=0.6, s=10)
    plt.title(title)
    plt.colorbar(scatter, label='Cluster Label')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()

# 3. 主执行流程

# A. 加载数据
X_train, X_test, y_train, y_test = load_and_filter_mnist()

# 实验 1: 原始数据直接聚类 (基准实验)
print("\n>>> 开始实验 1: 原始高维数据 (784维) 直接聚类")
start_time = time.time()

kmeans_raw = SimpleKMeans(k=3, max_iters=300)
kmeans_raw.fit(X_train)
y_pred_raw = kmeans_raw.predict(X_test)

print(f"耗时: {time.time() - start_time:.2f} 秒")
# 评估
y_pred_raw_mapped = evaluate_clustering(y_test, y_pred_raw, "原始 K-Means")

# 为了可视化，我们必须用 PCA 把 784维 降到 2维
pca_vis = PCA(n_components=2)
X_test_pca_vis = pca_vis.fit_transform(X_test)
plot_results(X_test_pca_vis, y_pred_raw_mapped, "Experiment 1: Raw K-Means Result (Visualized by PCA)")

# 实验 2: 优化尝试 (PCA 降维 + 特征提取)
print("\n>>> 开始实验 2: 优化方案 (PCA降维 -> K-Means)")
print("优化思路: MNIST图像在高维空间稀疏且包含噪声，欧氏距离在高维下失效(维数灾难)。先降维提取主成分。")

# 1. 使用 PCA 将 784 维降到 30 维 (保留约80-90%的方差信息，去除噪声)
pca = PCA(n_components=0.90)  # 保留90%的方差
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
print(f"PCA 降维后特征数量: {X_train_pca.shape[1]}")

start_time = time.time()

# 2. 在降维后的数据上训练 K-Means
kmeans_opt = SimpleKMeans(k=3, max_iters=300)
kmeans_opt.fit(X_train_pca)
y_pred_opt = kmeans_opt.predict(X_test_pca)

print(f"耗时: {time.time() - start_time:.2f} 秒")
# 评估
y_pred_opt_mapped = evaluate_clustering(y_test, y_pred_opt, "优化后 (PCA + K-Means)")

# 可视化 (直接使用PCA的前两个分量)
plot_results(X_test_pca[:, :2], y_pred_opt_mapped, "Experiment 2: Optimized K-Means (PCA Preprocessing)")


