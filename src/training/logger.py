import logging
import os
from datetime import datetime


class Logger:
    def __init__(self, log_dir="logs", experiment_name="base_model"):
        """
        Initializes the Logger for console, file, and TensorBoard logging.

        Args:
            log_dir (str): Directory to save log files and TensorBoard logs.
            experiment_name (str): Name of the experiment for organization.
        """
        # Create directories
        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # File logging setup
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file = os.path.join(self.log_dir, f"training_{current_time}.log")
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console logging setup
        self.console_logger = logging.getLogger()
        self.console_logger.addHandler(logging.StreamHandler())

    def log(self, message, level=logging.INFO):
        """
        Logs a message to both the console and the log file.

        Args:
            message (str): The message to log.
            level (int): Logging level (e.g., INFO, WARNING).
        """
        if level == logging.INFO:
            self.console_logger.info(message)
        elif level == logging.WARNING:
            self.console_logger.warning(message)
        elif level == logging.ERROR:
            self.console_logger.error(message)
