#  Univariate Linear Regression — From Scratch

Implemented **Univariate Linear Regression from scratch** using only NumPy — no sklearn models used. Every component including gradient descent, MSE loss, and gradient computation is written manually.

---

## About the Project

This project predicts **expected marks** based on **study hours** using a simple linear regression model built from the ground up.

- Dataset: Study Hours vs Expected Marks (100 samples)
- All math implemented manually — no `sklearn.linear_model`
- Visualizes both the regression line and the loss curve

---

##  Concepts Used

- Linear Equation: `y = mX + b`
- Mean Squared Error (MSE) Loss
- Gradient Descent Optimization
- MinMax Feature Scaling

---

## . Project Structure

```
├── Univariate_LR.py       # Main code
├── Data/
│   └── data.csv           # Dataset (Study Hours vs Marks)
└── README.md
```

---

## ⚙️ How It Works

### 1. Data Preprocessing
- Load CSV using NumPy
- Scale both X (hours) and y (marks) to range [0, 1] using MinMaxScaler

### 2. Core Functions

| Function | Purpose |
|---|---|
| `predict_y(X, m, b)` | Predicts output using `y = mX + b` |
| `gradient(X, y, m, b)` | Computes gradients `∂L/∂m` and `∂L/∂b` |
| `loss(X, y, m, b)` | Computes Mean Squared Error |
| `gradient_descent(...)` | Runs the training loop for `max_itr` iterations |

### 3. Hyperparameters

```python
learning_rate = 0.1
max_itr       = 1000
```

---

## 📊 Output

**Plot 1 — Regression Line on Scatter Data**

Shows how well the model fits the data after training.

**Plot 2 — Loss Curve**

Shows how the MSE loss decreases over 1000 iterations — confirms the model is learning.

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/RudraGor/univariate-linear-regression.git

# Install dependencies
pip install numpy matplotlib scikit-learn

# Run
python Univariate_LR.py
```

---

##  Dependencies

```
numpy
matplotlib
scikit-learn
```

---

##  Author

**Rudra Gor**  
[GitHub](https://github.com/RudraGor) | [LinkedIn](https://linkedin.com/in/RudraGor)
