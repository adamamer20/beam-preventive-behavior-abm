"""Normalized data writers for better data organization."""

import csv
import json
import queue
import threading
import time
from pathlib import Path
from typing import Any

from beam_abm.common.logging import get_logger

logger = get_logger(__name__)


class ModelRunWriter:
    """Writer for model run metadata."""

    def __init__(self, output_dir: str, filename: str = "model_runs.csv"):
        """Initialize the model run writer.

        Parameters
        ----------
        output_dir : str
            Directory to save the file
        filename : str, optional
            Name of the CSV file, by default "model_runs.csv"
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = self.output_dir / filename

        # Initialize CSV file with headers
        self.fieldnames = [
            "run_id",
            "model_name",
            "method",
            "start_time",
            "end_time",
            "total_input_tokens",
            "total_output_tokens",
            "total_tokens",
            "record_count",
            "avg_processing_time",
            "success_rate",
            "model_config",
        ]

        self._init_csv_file()
        logger.info(f"ModelRunWriter initialized: {self.filepath}")

    def _init_csv_file(self):
        """Initialize the CSV file with headers."""
        with open(self.filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def save_run_metadata(self, run_data: dict[str, Any]):
        """Save model run metadata.

        Parameters
        ----------
        run_data : Dict[str, Any]
            Run metadata to save
        """
        try:
            # Serialize model_config if it's a dict
            if "model_config" in run_data and isinstance(run_data["model_config"], dict):
                run_data["model_config"] = json.dumps(run_data["model_config"])

            with open(self.filepath, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writerow(run_data)

            logger.debug(f"Saved run metadata for {run_data.get('run_id', 'unknown')}")

        except Exception as e:
            logger.error(f"Error saving run metadata: {e}")


class PredictionWriter:
    """Non-blocking writer for individual predictions."""

    def __init__(self, output_dir: str, filename: str = "predictions.csv"):
        """Initialize the prediction writer.

        Parameters
        ----------
        output_dir : str
            Directory to save the file
        filename : str, optional
            Name of the CSV file, by default "predictions.csv"
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = self.output_dir / filename

        # Queue for predictions to be written
        self.prediction_queue = queue.Queue()

        # Threading control
        self._stop_event = threading.Event()
        self._writer_thread = None
        self._csv_file = None
        self._csv_writer = None
        self._rows_written = 0

        # Define fieldnames for predictions
        self.fieldnames = [
            "run_id",
            "record_id",
            "record_index",
            # Target columns (will be added dynamically)
            "institut_trust_vax_1",
            "institut_trust_vax_2",
            "institut_trust_vax_3",
            "institut_trust_vax_4",
            "institut_trust_vax_5",
            "institut_trust_vax_6",
            "institut_trust_vax_7",
            "institut_trust_vax_8",
            "opinion_1",
            "fiveC_1",
            "fiveC_2",
            "fiveC_3",
            "fiveC_4",
            "fiveC_5",
            "protective_behaviour_1",
            "protective_behaviour_2",
            "protective_behaviour_3",
            "protective_behaviour_4",
            # Token usage
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "processing_time",
            "validation_success",
            "validation_error",
            "attempt_number",
            "request_id",
            "timestamp",
        ]

        # Start the writer thread
        self._start_writer_thread()

        logger.info(f"PredictionWriter initialized: {self.filepath}")

    def _start_writer_thread(self):
        """Start the background writer thread."""
        self._writer_thread = threading.Thread(target=self._write_loop, daemon=True)
        self._writer_thread.start()
        logger.debug("Background prediction writer thread started")

    def _write_loop(self):
        """Background thread loop that writes predictions to CSV."""
        try:
            while not self._stop_event.is_set():
                try:
                    # Get prediction from queue with timeout
                    prediction = self.prediction_queue.get(timeout=1.0)

                    # Write the prediction
                    self._write_prediction(prediction)

                    # Mark task as done
                    self.prediction_queue.task_done()

                except queue.Empty:
                    # Timeout - continue loop to check stop event
                    continue
                except Exception as e:
                    logger.error(f"Error in prediction writer thread: {e}")

        except Exception as e:
            logger.error(f"Fatal error in prediction writer thread: {e}")
        finally:
            # Close CSV file if open
            if self._csv_file:
                self._csv_file.close()
                logger.debug("Prediction CSV file closed")

    def _write_prediction(self, prediction: dict[str, Any]):
        """Write a single prediction to the CSV file.

        Parameters
        ----------
        prediction : Dict[str, Any]
            Prediction data to write
        """
        try:
            # Initialize CSV writer if not done yet
            if self._csv_writer is None:
                self._init_csv_writer()

            # Write the prediction
            self._csv_writer.writerow(prediction)
            self._csv_file.flush()  # Ensure data is written to disk
            self._rows_written += 1

            if self._rows_written % 100 == 0:
                logger.debug(f"Written {self._rows_written} predictions")

        except Exception as e:
            logger.error(f"Error writing prediction: {e}")

    def _init_csv_writer(self):
        """Initialize the CSV writer with predefined fieldnames."""
        self._csv_file = open(self.filepath, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self.fieldnames)
        self._csv_writer.writeheader()
        logger.info(f"Prediction CSV file initialized with {len(self.fieldnames)} fields")

    def save_prediction(self, prediction: dict[str, Any]):
        """Save a prediction (non-blocking).

        Parameters
        ----------
        prediction : Dict[str, Any]
            Prediction data to save
        """
        if self._stop_event.is_set():
            logger.warning("Cannot save prediction - writer is stopped")
            return

        try:
            # Add timestamp if not present
            if "timestamp" not in prediction:
                prediction["timestamp"] = time.time()

            # Add to queue (non-blocking)
            self.prediction_queue.put(prediction, block=False)

        except queue.Full:
            logger.warning("Prediction queue is full - dropping prediction")
        except Exception as e:
            logger.error(f"Error queuing prediction: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get writer statistics.

        Returns
        -------
        Dict[str, Any]
            Writer statistics
        """
        return {
            "rows_written": self._rows_written,
            "queue_size": self.prediction_queue.qsize(),
            "is_alive": self._writer_thread.is_alive() if self._writer_thread else False,
            "filepath": str(self.filepath),
        }

    def close(self, timeout: float = 30.0):
        """Close the writer and wait for all predictions to be written.

        Parameters
        ----------
        timeout : float, optional
            Maximum time to wait for queue to empty, by default 30.0
        """
        logger.info("Closing PredictionWriter...")

        # Wait for queue to empty
        start_time = time.time()
        while not self.prediction_queue.empty() and (time.time() - start_time) < timeout:
            time.sleep(0.1)

        if not self.prediction_queue.empty():
            logger.warning(f"Prediction queue not empty after {timeout}s timeout - forcing close")

        # Stop the writer thread
        self._stop_event.set()

        # Wait for thread to finish
        if self._writer_thread and self._writer_thread.is_alive():
            self._writer_thread.join(timeout=5.0)
            if self._writer_thread.is_alive():
                logger.warning("Prediction writer thread did not stop gracefully")

        logger.info(f"PredictionWriter closed. Total predictions written: {self._rows_written}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class GroundTruthWriter:
    """Writer for ground truth data (one-time generation)."""

    def __init__(self, output_dir: str, filename: str = "ground_truth.csv"):
        """Initialize the ground truth writer.

        Parameters
        ----------
        output_dir : str
            Directory to save the file
        filename : str, optional
            Name of the CSV file, by default "ground_truth.csv"
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = self.output_dir / filename

        # Check if file already exists
        self.file_exists = self.filepath.exists()

        logger.info(f"GroundTruthWriter initialized: {self.filepath}")

    def save_ground_truth(self, ground_truth_data: list[dict[str, Any]]):
        """Save ground truth data.

        Parameters
        ----------
        ground_truth_data : List[Dict[str, Any]]
            List of ground truth records to save
        """
        if self.file_exists:
            logger.info("Ground truth file already exists - skipping")
            return

        try:
            if not ground_truth_data:
                logger.warning("No ground truth data provided")
                return

            # Get fieldnames from first record
            fieldnames = list(ground_truth_data[0].keys())

            with open(self.filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(ground_truth_data)

            logger.info(f"Saved {len(ground_truth_data)} ground truth records")
            self.file_exists = True

        except Exception as e:
            logger.error(f"Error saving ground truth data: {e}")


class NormalizedDataManager:
    """Manager for all normalized data writers."""

    def __init__(self, output_dir: str):
        """Initialize the normalized data manager.

        Parameters
        ----------
        output_dir : str
            Directory to save all normalized files
        """
        self.output_dir = output_dir

        # Initialize writers
        self.model_run_writer = ModelRunWriter(output_dir, "model_runs.csv")
        self.prediction_writer = PredictionWriter(output_dir, "predictions.csv")
        self.ground_truth_writer = GroundTruthWriter(output_dir, "ground_truth.csv")

        logger.info(f"NormalizedDataManager initialized in: {output_dir}")

    def save_run_metadata(self, run_data: dict[str, Any]):
        """Save model run metadata."""
        self.model_run_writer.save_run_metadata(run_data)

    def save_prediction(self, prediction: dict[str, Any]):
        """Save a prediction."""
        self.prediction_writer.save_prediction(prediction)

    def save_ground_truth(self, ground_truth_data: list[dict[str, Any]]):
        """Save ground truth data."""
        self.ground_truth_writer.save_ground_truth(ground_truth_data)

    def get_stats(self) -> dict[str, Any]:
        """Get statistics from all writers."""
        return {
            "model_runs": {"filepath": str(self.model_run_writer.filepath)},
            "predictions": self.prediction_writer.get_stats(),
            "ground_truth": {"filepath": str(self.ground_truth_writer.filepath)},
        }

    def close(self, timeout: float = 30.0):
        """Close all writers."""
        logger.info("Closing NormalizedDataManager...")
        self.prediction_writer.close(timeout)
        logger.info("NormalizedDataManager closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
