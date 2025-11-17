import numpy as np
import matplotlib.pyplot as plt

# ---- dữ liệu cũ ----
np.random.seed(0)
N = 200
X = np.random.randn(N, 2)
y = (X[:,0]**2 + X[:,1]**2 > 1).astype(int)  # tạo dữ liệu khó hơn một chút

plt.scatter(X[:,0], X[:,1], c=y, cmap='coolwarm')
plt.title("Dữ liệu 2D - Nonlinear")
plt.show()

# ---- khởi tạo mô hình ----
input_dim = 2
hidden_dim = 4
output_dim = 1

# khởi tạo ngẫu nhiên
W1 = np.random.randn(input_dim, hidden_dim)
b1 = np.zeros((1, hidden_dim))
W2 = np.random.randn(hidden_dim, output_dim)
b2 = np.zeros((1, output_dim))

print("W1 shape:", W1.shape)
print("W2 shape:", W2.shape)

# Hàm kích hoạt ReLU và Sigmoid
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Forward pass
def forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1        # (N,2) @ (2,4) -> (N,4)
    a1 = relu(z1)           # ReLU tạo phi tuyến
    z2 = a1 @ W2 + b2       # (N,4) @ (4,1) -> (N,1)
    y_hat = sigmoid(z2)     # Xác suất lớp 1
    return z1, a1, z2, y_hat

