Polynomial Regression (Basic understanding)

Fit y = w0 + w1*x + w2*x^2 + ... + wd*x^d
On polynomial features [1, x, x^2, ..., x^d].
Solve with least squares

import numpy as np
import matplotlib.pyplot as plt

#Noisy quadratic
np.random.seed(0)
x = np.linspace(-3, 3, 50)
y = 2 * x**2 - 3 * x + 1 + np.random.randn(50) * 2

#1
degree = 2
X = np.vander(x, degree + 1, increasing=True)   # columns: 1, x, x^2
w = np.linalg.lstsq(X, y, rcond=None)[0]         # least squares weights
y_pred = X @ w

print("weights (w0, w1, w2):", w.round(3))
print("MSE:", np.mean((y - y_pred) ** 2).round(3))

#Plotted
plt.scatter(x, y, label="data")
plt.plot(x, y_pred, color="red", label=f"degree {degree} fit")
plt.legend()
plt.show()
