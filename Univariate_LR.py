import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ─────────────────────────────────────────
# Data Preprocessing
# ─────────────────────────────────────────

data = np.genfromtxt("Data/data.csv", delimiter=",")

X = data[:, 0]  # Feature (Study Hours)
y = data[:, 1]  # Label   (Expected Marks)

# Separate scalers for X and y (good practice)
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

scaled_X = scaler_X.fit_transform(np.reshape(X, (X.shape[0], 1)))
scaled_y = scaler_y.fit_transform(np.reshape(y, (y.shape[0], 1)))

scaled_X = np.reshape(scaled_X, scaled_X.shape[0])
scaled_y = np.reshape(scaled_y, scaled_y.shape[0])

# ─────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────

learning_rate = 0.1
max_itr       = 1000

# ─────────────────────────────────────────
# Linear Regression Logic
# ─────────────────────────────────────────

def predict_y(X, m, b):
    """Calculate predicted y using y = mX + b"""
    return m * X + b


def gradient(X, y, m, b):
    """Calculate gradients of MSE loss w.r.t m and b"""
    y_hat = predict_y(X, m, b)
    dm    = np.average((y_hat - y) * X)   # ∂L/∂m
    db    = np.average(y_hat - y)          # ∂L/∂b
    return dm, db


def loss(X, y, m, b):
    """Mean Squared Error loss"""
    y_hat = predict_y(X, m, b)
    return np.average(np.square(y - y_hat))


def gradient_descent(X, y, learning_rate, max_itr):
    """Run gradient descent for max_itr iterations"""
    m      = 0.0
    b      = 0.0
    losses = []

    for i in range(max_itr):
        dm, db      = gradient(X, y, m, b)
        m           = m - learning_rate * dm
        b           = b - learning_rate * db
        loss_value  = loss(X, y, m, b)
        losses.append(loss_value)

    return m, b, losses, loss_value


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────


m, b, losses, final_loss = gradient_descent(
    scaled_X, scaled_y, learning_rate, max_itr
)

print("Slope (m)     :", m)
print("Intercept (b) :", b)
print("Final Loss    :", final_loss)

# ── Plot 1: Scatter + Regression Line ──
plt.figure(figsize=(8, 5))
plt.title("Linear Regression — Expected Marks vs Study Hours", fontsize=14)
plt.xlabel("Study Hours (Scaled)", fontsize=12)
plt.ylabel("Expected Marks (Scaled)", fontsize=12)

colors = np.random.rand(len(scaled_X), 3)
sizes  = np.random.randint(20, 500, size=len(scaled_X))
plt.scatter(scaled_X, scaled_y, c=colors, s=sizes, alpha=0.6, label="Data Points")

yy = predict_y(scaled_X, m, b)
plt.plot(scaled_X, yy, c='red', linewidth=2, label="Regression Line")
plt.legend()
plt.tight_layout()
plt.show()

    # ── Plot 2: Loss Curve ──
plt.figure(figsize=(8, 5))
plt.title("Loss over Iterations", fontsize=14)
plt.xlabel("Number of Iterations", fontsize=12)
plt.ylabel("Loss (MSE)", fontsize=12)
plt.plot(losses, c='blue', linewidth=2)
plt.tight_layout()
plt.show()