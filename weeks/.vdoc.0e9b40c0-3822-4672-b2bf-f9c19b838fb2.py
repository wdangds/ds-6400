# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import rdata 
import urllib.request 

from pathlib import Path 

from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV, ElasticNetCV, lasso_path

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score

from sklearn.metrics import mean_squared_error

url = "https://raw.githubusercontent.com/cran/hdi/master/data/riboflavin.RData"

data_path = Path("riboflavin.RData")

if not data_path.exists():
    urllib.request.urlretrieve(url, data_path)

# Preserve the unusual R data frame as a dictionary. Its `x` field is a
# 71 x 4088 matrix, which cannot be represented as one pandas DataFrame column.
def preserve_r_container(obj, _attrs):
    return obj


data = rdata.read_rda(data_path,
    constructor_dict={
        "data.frame": preserve_r_container,
        "AsIs": preserve_r_container,
    }
)

riboflavin = data["riboflavin"]

X = np.asarray(riboflavin["x"], dtype=float)
y = np.asarray(riboflavin["y"], dtype=float).ravel()

print("X shape:", X.shape)
print("y shape:", y.shape)
#
#
#
#
#
cv = KFold(n_splits=10, shuffle=True, random_state=6400)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
ols = LinearRegression()
ols.fit(X, y)

print("Number of observations:", X.shape[0])
print("Number of predictors:", X.shape[1])

print("Rank of centered X:", ols.rank_)
print("Number of coefficients:", len(ols.coef_))
print("Number of NA coefficients:", np.isnan(ols.coef_).sum())

print(
    "Number of non-unique coefficient directions:",
    X.shape[1] - ols.rank_
)

print(
    "Training MSE:",
    mean_squared_error(y, ols.predict(X))
)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
ridge_lambdas = np.logspace(-3, 6, 60)

ridge_pipeline = Pipeline([("scaler", StandardScaler()),
                            ("ridge", Ridge())])

ridge_cv = GridSearchCV(estimator = ridge_pipeline,
                        param_grid = {"ridge__alpha": ridge_lambdas},
                        cv = cv, 
                        scoring="neg_mean_squared_error",
                        refit=True)
ridge_cv.fit(X, y)

# minimum CV-MSE 
ridge_mse = -ridge_cv.cv_results_["mean_test_score"]
idx_min = np.argmin(ridge_mse)
ridge_lambda_min = ridge_lambdas[idx_min]
ridge_mse_min = ridge_mse[idx_min]

# standard error across the 10 validation folds 
ridge_fold_mse = np.array([-ridge_cv.cv_results_[f"split{i}_test_score"] for i in range(cv.get_n_splits())])

ridge_se = ridge_fold_mse.std(axis=0, ddof=1) / np.sqrt(10)

# one standard error: choose the largest lambda within 1 SE of minimum 
threshold = ridge_mse_min + ridge_se[idx_min]
eligible = np.where(ridge_mse <= threshold)[0]
idx_1se = eligible[-1]

ridge_lambda_1se = ridge_lambdas[idx_1se]

print("lambda.min:", ridge_lambda_min)
print("lambda.1se:", ridge_lambda_1se)
print("Minimum CV-MSE:", ridge_mse_min)
#
#
#
#
#
#
plt.figure(figsize=(8,5))

plt.errorbar(np.log(ridge_lambdas), ridge_mse, yerr=ridge_se, fmt="-o", markersize=3, capsize=2, label="10-fold CV-MSE")
plt.axvline(np.log(ridge_lambda_min), color = "red", linestyle="--", label = "lambda.min")
plt.axvline(np.log(ridge_lambda_1se), color= "green", linestyle="--", label="lambda.1se")
plt.xlabel(r"$\log(\lambda)$")
plt.ylabel("Cross validation MSE")
plt.legend()
plt.show()
#
#
#
#
#
#
#
#
#
#
#
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

ridge_coefs = []

for lam in ridge_lambdas:
    model = Ridge(alpha=lam)
    model.fit(X_scaled, y)
    ridge_coefs.append(model.coef_)

ridge_coefs = np.array(ridge_coefs)
print(ridge_coefs.shape)
#
#
#
plt.figure(figsize=(8,5))

plt.plot(-np.log(ridge_lambdas), ridge_coefs, linewidth=0.5, alpha=0.25)
plt.axvline(-np.log(ridge_lambda_min), color = "red", linestyle="--", label = "lambda.min")
plt.xlabel(r"$-\log(\lambda)$")
plt.ylabel("Standardize coefficient values")
plt.title("Ridge regression coefficient paths")
plt.legend()
plt.show()

#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
lasso_pipeline = Pipeline([("scaler", StandardScaler()),
                            ("lasso", LassoCV(
                                cv = cv, 
                                max_iter = 20000, 
                                tol=1e-4,
                                random_state=6400
                            ))])
lasso_pipeline.fit(X, y)
lasso_model = lasso_pipeline.named_steps["lasso"]
lasso_lambda_min = lasso_model.alpha_
lasso_mse_curve = lasso_model.mse_path_.mean(axis=1)
lasso_mse_min = lasso_mse_curve.min()

# number of non-zero coeffcients 
lasso_nonzero = np.count_nonzero(np.abs(lasso_model.coef_) > 1e-8)

print("Optimal lambda:", lasso_lambda_min)
print("Minimum CV-MSE:", lasso_mse_min)
print("Number of selected genes:", lasso_nonzero)
#
#
#
#
#
lasso_se =  (lasso_model.mse_path_.std(axis=1, ddof=1) / np.sqrt(10))

plt.figure(figsize=(8,5))
plt.errorbar(np.log(lasso_model.alphas_), lasso_mse_curve, yerr=lasso_se, fmt="-o", markersize=3, capsize=2)
plt.axvline(np.log(lasso_lambda_min), color = "red", linestyle="--", label="Optimal lambda")
plt.xlabel(r"$\log(\lambda)$")
plt.ylabel("Cross-validation MSE")
plt.title("Lasso  Regression: 10-fold CV")
plt.legend()
plt.show()
#
#
#
lasso_alphas , lasso_coefs, _ = lasso_path(X_scaled, y-y.mean(), alphas=lasso_model.alphas_, max_iter=20000)
plt.figure(figsize=(10,6))
plt.plot(-np.log(lasso_alphas), lasso_coefs.T, linewidth=0.5, alpha=0.5)
plt.axvline(-np.log(lasso_lambda_min), color="red", linestyle="--", label="Optimal lambda")
plt.xlabel(r"$-\log(\lambda)$")
plt.ylabel("Standardized coefficient values")
plt.title("Lasso Regression Coefficient Paths")
plt.legend()
plt.show()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
mixing_values = [0.1, 0.25, 0.5, 0.75, 0.9]

enet_pipeline = Pipeline([("scaler", StandardScaler()),
                            ("enet", ElasticNetCV(
                                l1_ratio=mixing_values,
                                cv = cv, 
                                max_iter = 20000,
                                tol = 1e-4,
                                random_state = 6400
                            ))])
enet_pipeline.fit(X, y)
enet_model = enet_pipeline.named_steps["enet"]
print("Best mixing parameter:", enet_model.l1_ratio_)
print("Best regularization strength:", enet_model.alpha_)
#
#
#
#
#
results_enet = []

for i, mixing in enumerate(mixing_values):
    mse_curve = enet_model.mse_path_[i].mean(axis=1)
    best_idx = np.argmin(mse_curve)
    best_lambda = enet_model.alphas_[i, best_idx]
    best_mse = mse_curve[best_idx]

    results_enet.append({"Mixing alpha": mixing,
                        "Best lambda": best_lambda,
                        "Minimum CV-MSE": best_mse})
enet_table = pd.DataFrame(results_enet)
print(enet_table.to_string(index=False))
#
#
#
best_row = enet_table.loc[enet_table["Minimum CV-MSE"].idxmin()]
print(best_row)
#
#
#
enet_nonzero = np.count_nonzero(np.abs(enet_model.coef_) > 1e-8)
feature_comparison = pd.DataFrame({"Method": ["Lasso", "Elastic Net"],
                                    "Selected features": [lasso_nonzero, enet_nonzero]})
print(feature_comparison)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
max_components = min(len(train_idx) - 1 for train_idx, _ in cv.split(X))
component_grid = np.arange(1, max_componets + 1)
pcr_fold_mse = []

for train_idx, test_idx in cv.split(X):
    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    # fit scaler on training data only 
    scaler_fold = StandardScaler()
    X_train_s = scaler_fold.fit_transform(X_train)
    X_test_s = scaler_fold.transform(X_test)

    # SVD of standardized training predictors 
    U, s, Vt = np.linalg.svd(X_train_s, full_matrices = False)
    # component-space OLS coefficients
    y_mean = y_train.mean()
    gamma = np.divide(U.T @ ( y_train - y_mean), s, out=np.zeros_like(s), where=s>1e-10)
    # contributions of individual PCs to predictions 
    test_scores = x_test_s @ Vt.T
    contributions = test_scores * gamma 
    # predictions for model with 1, 2, ..., m PCs
    predictions = (y_mean + np.cumsum(contributions[:, :max_components], axis=1))
    fold_errors = np.mean((y_test[:, None] - predictions)**2, axis=0)
    pcr_fold_mse.append(fold_errors)

pcr_fold_mse = np.array(pcr_fold_mse)
pcr_mse_curve = pcr_fold_mse.mean(axis=0)
pcr_mse_se = (pcr_fold_mse.std(axis=0, ddof=1)/np.sqrt(10))
best_idx = np.argmin(pcr_mse_curve)
M_optimal = int(component_grid[best_idx])

#
#
#
