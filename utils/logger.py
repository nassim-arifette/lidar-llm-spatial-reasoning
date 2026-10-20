import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import tensorboard
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class Logger:
    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        use_tensorboard: bool = False,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
        log_level: int = logging.INFO,
    ):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        self.use_wandb = use_wandb and WANDB_AVAILABLE

        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)

        # Setup file logging
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(log_level)

        if not self.logger.handlers:
            # File handler for logging to file
            log_file = self.log_dir / f"{experiment_name}.log"
            file_handler = logging.FileHandler(log_file)
            file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_format)
            self.logger.addHandler(file_handler)

            # Console handler for logging to console
            console_handler = logging.StreamHandler()
            console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(console_format)
            self.logger.addHandler(console_handler)

        # Setup TensorBoard logging
        if self.use_tensorboard:
            self.tb_writer = SummaryWriter(log_dir=os.path.join(self.log_dir, "tensorboard", experiment_name))

        # Setup Weights & Biases logging
        if self.use_wandb:
            if wandb_config is None:
                wandb_config = {}

            wandb.init(
                project=wandb_project or "ml_platform",
                name=experiment_name,
                config=wandb_config,
                dir=str(self.log_dir),
            )

        self.metrics_history = {}
        self.start_time = time.time()

        self.info(f"Logger initialized for experiment: {experiment_name}")
        if not TENSORBOARD_AVAILABLE and use_tensorboard:
            self.warning("TensorBoard requested but not available. Install with: pip install tensorboard")
        if not WANDB_AVAILABLE and use_wandb:
            self.warning("Weights & Biases requested but not available. Install with: pip install wandb")

    def log(self, metrics: Dict[str, Any], epoch: Optional[int] = None):
        """
        Log metrics to all enabled logging methods.

        Args:
            metrics: Dictionary of metrics to log
            epoch: Current epoch (optional)
        """
        # Add metrics to history
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append(value)

        # Add timestamp and epoch info
        log_dict = {
            "timestamp": time.time() - self.start_time,
            **metrics
        }
        if epoch is not None:
            log_dict["epoch"] = epoch

        # Log to file
        metrics_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in log_dict.items()])
        self.info(f"Metrics: {metrics_str}")

        # Log to TensorBoard
        if self.use_tensorboard and epoch is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(key, value, epoch)

        # Log to Weights & Biases
        if self.use_wandb:
            wandb.log(log_dict)

        # Save metrics to JSON file
        metrics_file = self.log_dir / f"{self.experiment_name}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

    def log_hyperparams(self, params: Dict[str, Any]):
        """Log hyperparameters to TensorBoard and Weights & Biases."""
        if self.use_tensorboard:
            self.tb_writer.add_hparams(params, {})

        if self.use_wandb:
            wandb.config.update(params)

        # Also log to file
        params_str = ", ".join([f"{k}={v}" for k, v in params.items()])
        self.info(f"Hyperparameters: {params_str}")

    def log_model_graph(self, model, input_shape):
        """Log model graph to TensorBoard."""
        if self.use_tensorboard:
            try:
                import torch
                device = next(model.parameters()).device
                dummy_input = torch.zeros(input_shape, device=device)
                self.tb_writer.add_graph(model, dummy_input)
            except Exception as e:
                self.warning(f"Failed to log model graph: {e}")

    def log_image(self, tag: str, image, step: Optional[int] = None):
        """Log image to TensorBoard and Weights & Biases."""
        if self.use_tensorboard:
            self.tb_writer.add_image(tag, image, step)

        if self.use_wandb:
            wandb.log({tag: wandb.Image(image)}, step=step)

    def log_figure(self, tag: str, figure, step: Optional[int] = None):
        """Log matplotlib figure to TensorBoard and Weights & Biases."""
        if self.use_tensorboard:
            self.tb_writer.add_figure(tag, figure, step)

        if self.use_wandb:
            wandb.log({tag: wandb.Image(figure)}, step=step)

    def save_artifact(self, artifact_path: str, name: str, type: str,
                      metadata: Optional[Dict[str, Any]] = None):
        """Save an artifact to Weights & Biases."""
        if self.use_wandb:
            artifact = wandb.Artifact(name=name, type=type, metadata=metadata)
            artifact.add_file(artifact_path)
            wandb.log_artifact(artifact)

    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)

    def close(self):
        """Close all loggers."""
        if self.use_tensorboard:
            self.tb_writer.close()

        if self.use_wandb:
            wandb.finish()

        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)
