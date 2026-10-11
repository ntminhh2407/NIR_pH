"""
    Reusable utilities for NIR pH machine-learning workflows.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR


__all__ = [
    "metrics_calculation",
    "svr_test_evaluate",
    "objective_wrapper",
    "calculate_and_plot_shap",
    "spectra_plot",
    "features_KDEplot",
    "pH_KDEplot",
    "StandardNormalVariate",
    "MultiplicativeScatterCorrection",
]


def svr_test_evaluate(X_train, X_test, y_train, y_test, log_C, log_gamma, epsilon):
    """Train an RBF SVR and return the negative test MSE for optimization."""
    model = SVR(
        kernel="rbf",
        C=10**log_C,
        gamma=10**log_gamma,
        epsilon=epsilon,
    )
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    return -mean_squared_error(y_test, y_test_pred)

# Tạo objective chứa kết quả train (-MSE) để truyền vào Bayesian Optimizer
def objective_wrapper(X_train, X_test, y_train, y_test):
    """Create a Bayesian Optimization objective for a train/test split."""

    def objective(log_C, log_gamma, epsilon):
        return svr_test_evaluate(
            X_train,
            X_test,
            y_train,
            y_test,
            log_C,
            log_gamma,
            epsilon,
        )

    return objective


def metrics_calculation(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    bar_chart_name="",
    scatter_plot_name="",
    save_barchart=False,
    save_plot=False,
):
    """Calculate metrics and plot training/testing evaluation results."""
    y_pred_test = model.predict(X_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)

    y_pred_train = model.predict(X_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    r2_train = r2_score(y_train, y_pred_train)

    print(f"--- {bar_chart_name} ---")
    print("Training RMSE:", round(rmse_train, 4))
    print("Testing RMSE:", round(rmse_test, 4))
    print("Training R²:", round(r2_train, 4))
    print("Testing R²:", round(r2_test, 4))

    metrics = ["RMSE", "R²"]
    training_values = [rmse_train, r2_train]
    testing_values = [rmse_test, r2_test]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(
        x - width / 2, training_values, width, label="Training", color="skyblue"
    )
    rects2 = ax.bar(
        x + width / 2, testing_values, width, label="Testing", color="lightgreen"
    )

    ax.set_ylabel("Value")
    ax.set_title(bar_chart_name)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.4f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    autolabel(rects1)
    autolabel(rects2)
    fig.tight_layout()
    if save_barchart:
        plt.savefig(f"{bar_chart_name}.png", bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_test, alpha=0.5)
    plt.xlabel("Actual pH")
    plt.ylabel("Predicted pH")
    plt.title(scatter_plot_name)
    plt.grid(True)
    min_val = min(y_test.min(), y_pred_test.min())
    max_val = max(y_test.max(), y_pred_test.max())
    plt.plot([min_val, max_val], [min_val, max_val], "k--", lw=2)
    if save_plot:
        plt.savefig(f"{scatter_plot_name}.png", bbox_inches="tight")
    plt.show()


def calculate_and_plot_shap(
    model,
    X_train,
    X_test,
    feature_names,
    summary_title="",
    waterfall_title="",
    save_summary=0,
    save_waterfall=0,
):
    """Calculate and plot SHAP summary and waterfall plots."""
    background_data = shap.kmeans(X_train, 50)
    explainer = shap.KernelExplainer(model.predict, background_data)

    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    shap_values = explainer.shap_values(X_test_df)

    shap.summary_plot(shap_values, X_test_df, feature_names=feature_names)
    if save_summary:
        shap.summary_plot(shap_values, X_test_df, feature_names=feature_names, show=False)
        plt.savefig(f"{summary_title}.png", bbox_inches="tight")

    print("\n--- SHAP Waterfall Plot for the first test sample ---")
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=X_test_df.iloc[0],
            feature_names=feature_names,
        )
    )
    if save_waterfall:
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=X_test_df.iloc[0],
                feature_names=feature_names,
            ),
            show=False,
        )
        plt.savefig(f"{waterfall_title}.png", bbox_inches="tight")


def spectra_plot(df, plot_name="", save_fig=False):
    """Plot one spectrum per row of a DataFrame."""
    wavelengths = df.columns.values.astype(float)

    plt.figure(figsize=(8, 5))
    for i in range(len(df)):
        plt.plot(wavelengths, df.iloc[i], linewidth=0.3)

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Transmittance intensity / Absorbance")
    plt.title(f"{plot_name} ({df.shape[0]} samples)")
    plt.xticks(wavelengths, rotation=45)
    plt.grid(True)
    if save_fig:
        plt.savefig(f"{plot_name}.png")
    plt.show()


def features_KDEplot(df: pd.DataFrame, plot_name: str, save_fig: bool = False):
    """Plot KDE distributions for the numeric columns in a DataFrame."""
    cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    if not cols:
        print("No numerical columns to plot (excluding 'pH_Horiba').")
        return

    n_cols = 6
    n_rows = (len(cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = np.asarray(axes).flatten()

    for i, col in enumerate(cols):
        sns.histplot(data=df, x=col, kde=True, ax=axes[i], color="teal")
        axes[i].set_title(f"Distribution of {col}")
        axes[i].set_xlabel("")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    if save_fig:
        plt.savefig(f"{plot_name}.png")
    plt.show()


def pH_KDEplot(
    array,
    xlabel="pH",
    ylabel="Tần suất",
    title="",
    y_interval=0.2,
    save_fig=0,
):
    """Plot the KDE distribution of pH values with a histogram."""
    plt.figure(figsize=(10, 6))
    sns.histplot(array, kde=True, bins=20, color="skyblue")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.7)

    min_val = array.min()
    max_val = array.max()
    plt.xticks(
        np.arange(np.floor(min_val), np.ceil(max_val) + y_interval, y_interval),
        rotation=45,
    )

    plt.tight_layout()
    if save_fig:
        filename = title if title else "pH_KDE_Distribution"
        plt.savefig(f"{filename}.png")
    plt.show()


class StandardNormalVariate(BaseEstimator, TransformerMixin):
    """Apply standard normal variate preprocessing independently per row."""

    def __init__(self, ddof=1):
        self.ddof = ddof

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_arr = np.asarray(X)
        mean = np.mean(X_arr, axis=1, keepdims=True)
        std = np.std(X_arr, axis=1, ddof=self.ddof, keepdims=True)
        return (X_arr - mean) / std


class MultiplicativeScatterCorrection(BaseEstimator, TransformerMixin):
    """Apply multiplicative scatter correction using the fitted mean spectrum."""

    def __init__(self):
        self.reference_ = None

    def fit(self, X, y=None):
        X_arr = np.asarray(X)
        self.reference_ = np.mean(X_arr, axis=0)
        return self

    def transform(self, X):
        X_arr = np.asarray(X)
        X_msc = np.zeros_like(X_arr)
        for i in range(X_arr.shape[0]):
            if self.reference_ is None:
                raise RuntimeError("The transformer must be fitted before calling transform.")

            reference = np.asarray(self.reference_, dtype=float)
            spectrum = np.asarray(X_arr[i, :], dtype=float)
            fit = np.polyfit(reference, spectrum, 1)
            b, a = fit[0], fit[1]
            X_msc[i, :] = (X_arr[i, :] - a) / b
        return X_msc
