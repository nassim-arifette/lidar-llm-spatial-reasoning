import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Union, Tuple
from sklearn import metrics as skmetrics
import io
import itertools
from pathlib import Path
import pandas as pd


class Visualizer:
    """
    Utility class for creating various visualizations for machine learning models.
    """
    def __init__(self, save_dir: Optional[str] = None):
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # Set default style
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (10, 6)
        plt.rcParams["font.size"] = 12

    def plot_training_history(
        self,
        metrics: Dict[str, List],
        figsize: Tuple[int, int] = (12, 8),
        save_as: Optional[str] = None
    ):
        """
        Plot training metrics history.

        Args:
            metrics: Dictionary with metric names as keys and lists of values as values
            figsize: Figure size
            save_as: Filename to save the plot

        Returns:
            matplotlib figure
        """
        plt.figure(figsize=figsize)

        # Group related metrics (e.g. train_loss and val_loss)
        metric_groups = {}
        for key in metrics.keys():
            base_name = key.split('_')[0] if '_' in key else key
            if base_name not in metric_groups:
                metric_groups[base_name] = []
            metric_groups[base_name].append(key)

        # Create subplots for each metric group
        n_groups = len(metric_groups)
        fig, axes = plt.subplots(n_groups, 1, figsize=figsize, sharex=True)

        # Ensure axes is always a list even for a single subplot
        if n_groups == 1:
            axes = [axes]

        for i, (group_name, metric_keys) in enumerate(metric_groups.items()):
            ax = axes[i]
            for key in metric_keys:
                epochs = range(1, len(metrics[key]) + 1)
                ax.plot(epochs, metrics[key], label=key)

            ax.set_title(f"{group_name.capitalize()} metrics")
            ax.set_ylabel(group_name)
            ax.legend()
            ax.grid(True)

        # Set x-axis label for the bottom subplot
        axes[-1].set_xlabel("Epochs")

        plt.tight_layout()

        if save_as and self.save_dir:
            plt.savefig(self.save_dir / save_as, bbox_inches='tight', dpi=300)

        return fig

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str],
        normalize: bool = False,
        figsize: Tuple[int, int] = (10, 8),
        cmap: str = "Blues",
        save_as: Optional[str] = None
    ):
        """
        Plot confusion matrix.

        Args:
            cm: Confusion matrix
            class_names: List of class names
            normalize: Whether to normalize the confusion matrix
            figsize: Figure size
            cmap: Colormap
            save_as: Filename to save the plot

        Returns:
            matplotlib figure
        """
        if normalize:
            cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)

        # Show all ticks
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=class_names,
            yticklabels=class_names,
            ylabel='True label',
            xlabel='Predicted label'
        )

        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        fig.tight_layout()

        if save_as and self.save_dir:
            plt.savefig(self.save_dir / save_as, bbox_inches='tight', dpi=300)

        return fig

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        class_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (10, 8),
        save_as: Optional[str] = None
    ):
        """
        Plot ROC curve for binary or multiclass classification.

        Args:
            y_true: True labels
            y_score: Predicted scores/probabilities
            class_names: List of class names
            figsize: Figure size
            save_as: Filename to save the plot

        Returns:
            matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Check if binary or multiclass
        if len(y_score.shape) == 1 or y_score.shape[1] <= 2:
            # Binary classification
            fpr, tpr, _ = skmetrics.roc_curve(y_true, y_score if len(y_score.shape) == 1 else y_score[:, 1])
            roc_auc = skmetrics.auc(fpr, tpr)

            ax.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic (ROC) Curve')
            ax.legend(loc="lower right")
        else:
            # Multiclass classification
            n_classes = y_score.shape[1]

            if class_names is None:
                class_names = [f'Class {i}' for i in range(n_classes)]

            # Compute ROC curve and ROC area for each class
            fpr = {}
            tpr = {}
            roc_auc = {}

            # Binarize the labels for one-vs-rest approach
            y_true_bin = skmetrics.label_binarize(y_true, classes=range(n_classes))

            for i in range(n_classes):
                fpr[i], tpr[i], _ = skmetrics.roc_curve(y_true_bin[:, i], y_score[:, i])
                roc_auc[i] = skmetrics.auc(fpr[i], tpr[i])
                ax.plot(fpr[i], tpr[i], label=f'{class_names[i]} (area = {roc_auc[i]:.2f})')

            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic (ROC) Curve - One-vs-Rest')
            ax.legend(loc="lower right")

        if save_as and self.save_dir:
            plt.savefig(self.save_dir / save_as, bbox_inches='tight', dpi=300)

        return fig

    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        class_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (10, 8),
        save_as: Optional[str] = None
    ):
        """
        Plot Precision-Recall curve for binary or multiclass classification.

        Args:
            y_true: True labels
            y_score: Predicted scores/probabilities
            class_names: List of class names
            figsize: Figure size
            save_as: Filename to save the plot

        Returns:
            matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Check if binary or multiclass
        if len(y_score.shape) == 1 or y_score.shape[1] <= 2:
            # Binary classification
            precision, recall, _ = skmetrics.precision_recall_curve(
                y_true,
                y_score if len(y_score.shape) == 1 else y_score[:, 1]
            )
            avg_precision = skmetrics.average_precision_score(
                y_true,
                y_score if len(y_score.shape) == 1 else y_score[:, 1]
            )

            ax.plot(recall, precision, label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall Curve')
            ax.legend(loc="lower left")
        else:
            # Multiclass classification
            n_classes = y_score.shape[1]

            if class_names is None:
                class_names = [f'Class {i}' for i in range(n_classes)]

            # Compute Precision-Recall curve and average precision for each class
            precision = {}
            recall = {}
            avg_precision = {}

            # Binarize the labels for one-vs-rest approach
            y_true_bin = skmetrics.label_binarize(y_true, classes=range(n_classes))

            for i in range(n_classes):
                precision[i], recall[i], _ = skmetrics.precision_recall_curve(y_true_bin[:, i], y_score[:, i])
                avg_precision[i] = skmetrics.average_precision_score(y_true_bin[:, i], y_score[:, i])
                ax.plot(recall[i], precision[i], label=f'{class_names[i]} (AP = {avg_precision[i]:.2f})')

            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall Curve - One-vs-Rest')
            ax.legend(loc="lower left")

        if save_as and self.save_dir:
            plt.savefig(self.save_dir / save_as, bbox_inches='tight', dpi=300)

        return fig

    def plot_feature_importance(
        self,
        feature_importance: np.ndarray,
        feature_names: List[str],
        top_n: Optional[int] = None,
        figsize: Tuple[int, int] = (12, 8),
        save_as: Optional[str] = None
    ):
        """
        Plot feature importance.

        Args:
            feature_importance: Feature importance values
            feature_names: Feature names
            top_n: Number of top features to display
            figsize: Figure size
            save_as: Filename to save the plot

        Returns:
            matplotlib figure
        """
        # Create a DataFrame for easier sorting and plotting
        df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        })

        # Sort by importance
        df = df.sort_values('Importance', ascending=False)

        # Take top N features if specified
        if top_n:
            df = df.head(top_n)

        fig, ax = plt.subplots(figsize=figsize)

        # Plot horizontal bar chart
        sns.barplot(x='Importance', y='Feature', data=df, ax=ax)

        ax.set_title('Feature Importance')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')

        plt.tight_layout()

        if save_as and self.save_dir:
            plt.savefig(self.save_dir / save_as, bbox_inches='tight', dpi=300)

        return fig

    def plot_distribution(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
        feature_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (15, 10),
        n_cols: int = 3,
        save_as: Optional[str] = None
    ):
        """
        Plot distribution of features.

        Args:
            data: Feature data (n_samples, n_features)
            labels: Class labels
            class_names: List of class names
            feature_names: List of feature names
            figsize: Figure size
            n_cols: Number of columns in the grid
            save_as: Filename to save the plot

        Returns:
            matplotlib figure
        """
        n_features = data.shape[1]

        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(n_features)]

        # Calculate number of rows needed
        n_rows = int(np.ceil(n_features / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()

        for i in range(n_features):
            ax = axes[i]

            if labels is not None:
                # Plot distribution by class
                for cls in np.unique(labels):
                    cls_name = class_names[cls] if class_names else f'Class {cls}'
                    sns.kdeplot(data[labels == cls, i], ax=ax, label=cls_name)
                ax.legend()
            else:
                # Plot overall distribution
                sns.histplot(data[:, i], ax=ax, kde=True)

            ax.set_title(feature_names[i])
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')

        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        if save_as and self.save_dir:
            plt.savefig(self.save_dir / save_as, bbox_inches='tight', dpi=300)

        return fig

    def plot_correlation_matrix(
        self,
        data: np.ndarray,
        feature_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 10),
        save_as: Optional[str] = None
    ):
        """
        Plot correlation matrix of features.

        Args:
            data: Feature data (n_samples, n_features)
            feature_names: List of feature names
            figsize: Figure size
            save_as: Filename to save the plot

        Returns:
            matplotlib figure
        """
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(data.T)

        n_features = data.shape[1]

        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(n_features)]

        fig, ax = plt.subplots(figsize=figsize)

        # Plot heatmap
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Correlation coefficient', rotation=-90, va="bottom")

        # Set ticks and labels
        ax.set_xticks(np.arange(n_features))
        ax.set_yticks(np.arange(n_features))
        ax.set_xticklabels(feature_names, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticklabels(feature_names)

        # Add text annotations
        for i in range(n_features):
            for j in range(n_features):
                text = ax.text(j, i, f"{corr_matrix[i, j]:.2f}",
                              ha="center", va="center", 
                              color="white" if abs(corr_matrix[i, j]) > 0.7 else "black")

        ax.set_title("Feature Correlation Matrix")
        fig.tight_layout()

        if save_as and self.save_dir:
            plt.savefig(self.save_dir / save_as, bbox_inches='tight', dpi=300)

        return fig

    def plot_learning_curve(
        self,
        train_sizes: np.ndarray,
        train_scores: np.ndarray,
        val_scores: np.ndarray,
        ylim: Optional[Tuple[float, float]] = None,
        figsize: Tuple[int, int] = (10, 6),
        save_as: Optional[str] = None
    ):
        """
        Plot learning curve.

        Args:
            train_sizes: Array of training sizes
            train_scores: 2D array of training scores for each training size
            val_scores: 2D array of validation scores for each training size
            ylim: Y-axis limits
            figsize: Figure size
            save_as: Filename to save the plot

        Returns:
            matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        # Plot learning curves
        ax.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')

        ax.plot(train_sizes, val_mean, 'o-', color='g', label='Validation score')
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='g')

        # Set plot title and labels
        ax.set_title('Learning Curve')
        ax.set_xlabel('Training examples')
        ax.set_ylabel('Score')

        # Set y-axis limits if provided
        if ylim:
            ax.set_ylim(*ylim)

        ax.legend(loc='best')
        ax.grid(True)

        if save_as and self.save_dir:
            plt.savefig(self.save_dir / save_as, bbox_inches='tight', dpi=300)

        return fig

    def plot_3d_data(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (10, 8),
        feature_names: Optional[List[str]] = None,
        save_as: Optional[str] = None
    ):
        """
        Plot 3D scatter plot of data.

        Args:
            X: Feature data (n_samples, n_features) where n_features >= 3
            y: Class labels
            class_names: List of class names
            figsize: Figure size
            feature_names: List of feature names
            save_as: Filename to save the plot

        Returns:
            matplotlib figure
        """
        if X.shape[1] < 3:
            raise ValueError("Input data must have at least 3 features for 3D plotting")

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(X.shape[1])]

        if y is not None:
            # Plot with class labels
            for i, cls in enumerate(np.unique(y)):
                cls_name = class_names[cls] if class_names else f'Class {cls}'
                ax.scatter(
                    X[y == cls, 0],
                    X[y == cls, 1],
                    X[y == cls, 2],
                    label=cls_name,
                    alpha=0.7
                )
            ax.legend()
        else:
            # Plot without class labels
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], alpha=0.7)

        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.set_zlabel(feature_names[2])
        ax.set_title('3D Feature Visualization')

        if save_as and self.save_dir:
            plt.savefig(self.save_dir / save_as, bbox_inches='tight', dpi=300)

        return fig

    def plot_model_comparison(
            self,
            model_names: List[str],
            metrics: Dict[str, List[float]],
            figsize: Tuple[int, int] = (12, 8),
            save_as: Optional[str] = None
    ):
        """
        Plot comparison of different models across multiple metrics.

        Args:
            model_names: List of model names
            metrics: Dictionary with metric names as keys and lists of values as values
                    (each list should have the same length as model_names)
            figsize: Figure size
            save_as: Filename to save the plot

        Returns:
            matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Number of metrics and models
        n_metrics = len(metrics)
        n_models = len(model_names)

        # Set width of bars and positions
        bar_width = 0.8 / n_metrics
        x = np.arange(n_models)

        # Plot bars for each metric
        for idx, (metric_name, values) in enumerate(metrics.items()):
            # Offset each metric's bars
            offset = (idx - n_metrics/2 + 0.5) * bar_width
            ax.bar(x + offset, values, bar_width, label=metric_name)

        # Customize plot
        ax.set_xlabel('Models')
        ax.set_ylabel('Metric Values')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()

        # Add grid for better readability
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()

        # Save if requested
        if save_as and self.save_dir:
            plt.savefig(self.save_dir / save_as, bbox_inches='tight', dpi=300)

        return fig
