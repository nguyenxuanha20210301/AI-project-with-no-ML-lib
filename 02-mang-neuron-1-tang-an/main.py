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

def compute_loss(y_hat, y):
    eps = 1e-8
    y = y.reshape(-1, 1)
    loss = -np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))
    return loss

# backward - muc dich: tim dao ham cua loss cho w1, b1, w2, b2
def backward(X, y, z1, a1, y_hat, W2):
    N = X.shape[0]
    y = y.reshape(-1, 1)

    # Gradient tại output
    dz2 = y_hat - y                       # (N,1)
    dW2 = a1.T @ dz2 / N                  # (4,N)@(N,1)->(4,1)
    db2 = np.mean(dz2, axis=0, keepdims=True)

    # Gradient tại hidden layer
    da1 = dz2 @ W2.T                      # (N,4)
    dz1 = da1 * (z1 > 0)                  # đạo hàm ReLU
    dW1 = X.T @ dz1 / N                   # (2,N)@(N,4)->(2,4)
    db1 = np.mean(dz1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2

learning_rate = 0.1
epochs = 2000

for epoch in range(epochs):
    # 1. Forward
    z1, a1, z2, y_hat = forward(X, W1, b1, W2, b2)
    
    # 2. Loss
    loss = compute_loss(y_hat, y)
    
    # 3. Backward
    dW1, db1, dW2, db2 = backward(X, y, z1, a1, y_hat, W2)
    
    # 4. Cập nhật tham số
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

def predict(X, W1, b1, W2, b2):
    _, _, _, y_hat = forward(X, W1, b1, W2, b2)
    return (y_hat >= 0.5).astype(int)

y_pred = predict(X, W1, b1, W2, b2)
acc = np.mean(y_pred.flatten() == y)
print("Độ chính xác:", acc)

# Vẽ decision boundary
x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
_, _, _, probs = forward(grid, W1, b1, W2, b2)
probs = probs.reshape(xx.shape)

plt.contourf(xx, yy, probs, levels=[0,0.5,1], alpha=0.3, cmap="coolwarm")
plt.scatter(X[:,0], X[:,1], c=y, cmap="coolwarm", edgecolor="k")
plt.title("Decision boundary - Neural Network 1 tầng ẩn")
plt.show()
