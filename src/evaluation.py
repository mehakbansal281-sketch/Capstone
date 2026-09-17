import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.ingestion import DataIngestionPipeline
from src.preprocessing import DataPreprocessor, NUMERICAL_FEATURES, TARGET_COLUMN

logger = logging.getLogger("EDAEvaluation")

# Set aesthetic plot style
plt.style.use("ggplot")
sns.set_theme(style="whitegrid")

class Visualizer:
    def __init__(self, data_path: str = "data/sales_data.csv", output_dir: str = "reports/plots"):
        self.data_path = data_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_all_plots(self):
        logger.info("Generating EDA & Model Comparison Visualizations...")
        ingestion = DataIngestionPipeline(self.data_path)
        df = ingestion.load_data()

        self.plot_correlation_heatmap(df)
        self.plot_distribution_plots(df)
        self.plot_missing_values(df)
        self.plot_business_insights(df)
        self.plot_model_comparison()
        self.plot_feature_importance()
        logger.info(f"All visualizations successfully saved to '{self.output_dir}'.")

    def plot_correlation_heatmap(self, df: pd.DataFrame):
        plt.figure(figsize=(10, 7))
        num_df = df[NUMERICAL_FEATURES + [TARGET_COLUMN]].copy().dropna()
        corr = num_df.corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5)
        plt.title("Correlation Heatmap of Numerical Features & Target", fontsize=14, fontweight="bold")
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, "correlation_heatmap.png")
        plt.savefig(filepath, dpi=300)
        plt.close()
        logger.info(f"Saved: {filepath}")

    def plot_distribution_plots(self, df: pd.DataFrame):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        sns.histplot(df[TARGET_COLUMN].dropna(), kde=True, ax=axes[0, 0], color="teal", bins=30)
        axes[0, 0].set_title("Distribution of Sales Amount ($)", fontweight="bold")

        sns.histplot(df["Unit_Price"].dropna(), kde=True, ax=axes[0, 1], color="darkorange", bins=30)
        axes[0, 1].set_title("Distribution of Unit Price ($)", fontweight="bold")

        sns.histplot(df["Marketing_Spend"].dropna(), kde=True, ax=axes[1, 0], color="purple", bins=30)
        axes[1, 0].set_title("Distribution of Marketing Spend ($)", fontweight="bold")

        sns.boxplot(x=df["Quantity_Ordered"].dropna(), ax=axes[1, 1], color="forestgreen")
        axes[1, 1].set_title("Boxplot of Quantity Ordered", fontweight="bold")

        plt.suptitle("Feature & Target Distribution Analysis", fontsize=16, fontweight="bold")
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, "distribution_plots.png")
        plt.savefig(filepath, dpi=300)
        plt.close()
        logger.info(f"Saved: {filepath}")

    def plot_missing_values(self, df: pd.DataFrame):
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        
        plt.figure(figsize=(8, 5))
        if len(missing) == 0:
            plt.text(0.5, 0.5, "No Missing Values Found", fontsize=14, ha="center")
        else:
            sns.barplot(x=missing.index, y=missing.values, palette="Reds_r")
            plt.ylabel("Missing Count", fontweight="bold")
            plt.title("Missing Value Analysis per Feature", fontsize=14, fontweight="bold")
            for i, v in enumerate(missing.values):
                plt.text(i, v + 2, str(v), ha="center", fontweight="bold")
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, "missing_value_analysis.png")
        plt.savefig(filepath, dpi=300)
        plt.close()
        logger.info(f"Saved: {filepath}")

    def plot_business_insights(self, df: pd.DataFrame):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        country_sales = df.groupby("Country")[TARGET_COLUMN].sum().sort_values(ascending=False)
        sns.barplot(x=country_sales.values, y=country_sales.index, ax=axes[0], palette="Blues_r")
        axes[0].set_title("Total Revenue by Country ($)", fontsize=14, fontweight="bold")
        axes[0].set_xlabel("Total Sales ($)")

        category_sales = df.groupby("Product_Category")[TARGET_COLUMN].mean().sort_values(ascending=False)
        sns.barplot(x=category_sales.values, y=category_sales.index, ax=axes[1], palette="Greens_r")
        axes[1].set_title("Average Transaction Value by Category ($)", fontsize=14, fontweight="bold")
        axes[1].set_xlabel("Average Sales ($)")

        plt.suptitle("Business Strategic Insights Dashboard", fontsize=16, fontweight="bold")
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, "business_insights.png")
        plt.savefig(filepath, dpi=300)
        plt.close()
        logger.info(f"Saved: {filepath}")

    def plot_model_comparison(self, metadata_path: str = "models/model_metadata.json"):
        if not os.path.exists(metadata_path):
            logger.warning(f"Metadata file '{metadata_path}' missing. Skipping model comparison plot.")
            return

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        results = metadata.get("evaluation_results", {})
        models = list(results.keys())
        rmse_scores = [results[m]["RMSE"] for m in models]
        r2_scores = [results[m]["R2"] for m in models]

        fig, ax1 = plt.subplots(figsize=(12, 6))

        color = "tab:red"
        ax1.set_xlabel("Model Architecture", fontweight="bold")
        ax1.set_ylabel("RMSE (Lower is better)", color=color, fontweight="bold")
        bars = ax1.bar(models, rmse_scores, color=color, alpha=0.6, width=0.4, align="center")
        ax1.tick_params(axis="y", labelcolor=color)
        plt.xticks(rotation=15, ha="right")

        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f"{height:.1f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

        ax2 = ax1.twinx()
        color = "tab:blue"
        ax2.set_ylabel("R² Score (Higher is better)", color=color, fontweight="bold")
        ax2.plot(models, r2_scores, color=color, marker="o", linewidth=2.5, markersize=8)
        ax2.tick_params(axis="y", labelcolor=color)
        ax2.set_ylim(-0.1, 1.1)

        plt.title("Baseline vs Advanced Machine Learning Models Performance Comparison", fontsize=14, fontweight="bold")
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, "baseline_vs_models_comparison.png")
        plt.savefig(filepath, dpi=300)
        plt.close()
        logger.info(f"Saved: {filepath}")

    def plot_feature_importance(self, model_path: str = "models/best_model.joblib",
                                preprocessor_path: str = "models/preprocessing_pipeline.joblib"):
        if not (os.path.exists(model_path) and os.path.exists(preprocessor_path)):
            logger.warning("Model or preprocessor missing. Skipping feature importance plot.")
            return

        best_model = joblib.load(model_path)
        preprocessor = DataPreprocessor.load(preprocessor_path)
        feature_names = preprocessor.get_feature_names()

        importances = None
        if hasattr(best_model, "feature_importances_"):
            importances = best_model.feature_importances_
        elif hasattr(best_model, "coef_"):
            importances = np.abs(best_model.coef_)

        if importances is None:
            logger.warning("Model does not provide feature importances/coefficients.")
            return

        fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        fi_df = fi_df.sort_values(by="Importance", ascending=False).head(10)

        plt.figure(figsize=(10, 6))
        sns.barplot(x="Importance", y="Feature", data=fi_df, palette="viridis")
        plt.title("Top 10 Feature Importance Analysis", fontsize=14, fontweight="bold")
        plt.xlabel("Relative Importance Score", fontweight="bold")
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, "feature_importance.png")
        plt.savefig(filepath, dpi=300)
        plt.close()
        logger.info(f"Saved: {filepath}")

if __name__ == "__main__":
    visualizer = Visualizer()
    visualizer.generate_all_plots()
