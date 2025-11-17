import numpy as np
import matplotlib.pyplot as plt

# Tạo dữ liệu ngẫu nhiên w
np.random.seed(0)
N = 200  # số điểm
X = np.random.randn(N, 2)

# Tạo nhãn theo một đường thẳng (để mô hình học)
y = (X[:,0] + X[:,1] > 0).astype(int)  # 1 nếu x+y>0, ngược lại 0

# Vẽ dữ liệu
plt.scatter(X[:,0], X[:,1], c=y, cmap="coolwarm")
plt.title("Dữ liệu phân loại 2D")
plt.show()


# Hàm sigmoid: biến số thực -> [0,1]
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Khởi tạo trọng số (weights) và bias ngẫu nhiên
W = np.random.randn(2, 1)   # 2 features -> vector (2x1)
b = 0.0                     # bias là 1 số

print("W ban đầu:", W.ravel())
print("b ban đầu:", b)

# Forward pass: từ X -> y_hat (xác suất thuộc lớp 1)
def forward(X, W, b):
    z = X @ W + b          # (N,2) @ (2,1) -> (N,1)
    y_hat = sigmoid(z)     # (N,1)
    return y_hat

# Hàm loss: Binary cross-entropy
def compute_loss(y_hat, y):
    eps = 1e-8
    y = y.reshape(-1, 1)  # đảm bảo shape (N,1)
    loss = -np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))
    return loss

# Chuyển y về dạng (N,1)
y_col = y.reshape(-1, 1)

learning_rate = 0.1
epochs = 1000

for epoch in range(epochs):
    # 1. Forward
    y_hat = forward(X, W, b)           # (N,1)

    # 2. Tính gradient
    error = y_hat - y_col              # (N,1)
    grad_W = X.T @ error / N           # (2,N) @ (N,1) -> (2,1)
    grad_b = np.mean(error)            # scalar

    # 3. Cập nhật tham số
    W -= learning_rate * grad_W
    b -= learning_rate * grad_b

    # 4. In loss mỗi 100 vòng
    if epoch % 100 == 0:
        loss = compute_loss(y_hat, y)
        print(f"Epoch {epoch}, loss = {loss:.4f}")

print("W sau khi học:", W.ravel())
print("b sau khi học:", b)


def predict(X, W, b):
    probs = forward(X, W, b)
    return (probs >= 0.5).astype(int)

y_pred = predict(X, W, b)
acc = np.mean(y_pred.flatten() == y)
print("Độ chính xác (accuracy):", acc)


# Vẽ decision boundary
x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200)
)

grid = np.c_[xx.ravel(), yy.ravel()]           # (200*200, 2)
probs = forward(grid, W, b).reshape(xx.shape)  # (200,200)

plt.contourf(xx, yy, probs, levels=[0, 0.5, 1], alpha=0.3, cmap="coolwarm")
plt.scatter(X[:,0], X[:,1], c=y, cmap="coolwarm", edgecolor="k")
plt.title("Decision boundary sau khi train")
plt.show()
