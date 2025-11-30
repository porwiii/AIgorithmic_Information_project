import numpy as np
import pandas as pd
import sklearn as sk
import scipy as sp
import matplotlib.pyplot as plt

from Model_selection_GA import ga_model_selection_mdl, evaluate_model_stats
from PolyReg_GA_prototype import ga_model_selection

np.random.seed(10)

# X = np.linspace(-3, 3, 100)
# y = 0.5 * X**3 - X + 2 + 0.3 * np.random.randn(len(X))
#
# best_model, best_mdl = ga_model_selection_mdl(X, y, mdl_type = "cmatrix_with_min1", mutation_prob=0.6, model_types=["linear", "ridge", "tree", "rf", "poly"])
# print("Best model:", best_model)
# print("MDL:", best_mdl)
#
# plt.plot(X, y)
# plt.show()
def make_sphere_data(n_samples=1000, dim=5, noise=0.0):
    X = np.random.uniform(-5, 5, size=(n_samples, dim))
    y_true = np.sum(X**2, axis=1)
    y = y_true + noise * np.random.randn(n_samples)
    return X, y

X, y = make_sphere_data(n_samples=1000, dim=5, noise=0.1)

best_bic, best_mdl_bic = ga_model_selection_mdl(
    X, y,
    generations=3,
    mdl_type="bic",
)

best_c, best_mdl_c = ga_model_selection_mdl(
    X, y,
    generations=3,
    mdl_type="cmatrix",
)

best_min1, best_mdl_min1 = ga_model_selection_mdl(
    X, y,
    generations=3,
    mdl_type="cmatrix_with_min1",
)

print("BIC best:", best_bic, "MDL_BIC:", best_mdl_bic)
print("Cmatrix best:", best_c, "MDL_C:", best_mdl_c)
print("Cmatrix modified best:", best_min1, "MDL_C:", best_mdl_min1)

stats = evaluate_model_stats(best_bic, X, y)
for k, v in stats.items(): print(f"{k}: {v}")

stats = evaluate_model_stats(best_c, X, y)
for k, v in stats.items(): print(f"{k}: {v}")

stats = evaluate_model_stats(best_min1, X, y)
for k, v in stats.items(): print(f"{k}: {v}")
#
#
#
# # Najlepszy osobnik: {'model_type': 'poly', 'degree': 3, 'n_knots': 4, 'max_depth': 5}
# MDL: -225.49772782541703

# Najlepszy osobnik: {'model_type': 'poly', 'degree': 3, 'n_knots': 7, 'max_depth': 2}
# MDL: 33.250933546472865 ten sam
