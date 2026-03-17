"""Non-blocking response writer for streaming LLM responses to disk."""

import csv
import queue
import threading
import time
from pathlib import Path
from typing import Any

from beam_abm.common.logging import get_logger

logger = get_logger(__name__)


class ResponseWriter:
    """Non-blocking writer for streaming LLM responses to CSV files."""

    def __init__(self, output_dir: str, filename: str = "responses.csv"):
        """Initialize the response writer.

        Parameters
        ----------
        output_dir : str
            Directory to save the response file
        filename : str, optional
            Name of the CSV file, by default "responses.csv"
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = self.output_dir / filename

        # Queue for responses to be written
        self.response_queue = queue.Queue()

        # Threading control
        self._stop_event = threading.Event()
        self._writer_thread = None
        self._csv_file = None
        self._csv_writer = None
        self._fieldnames: list[str] | None = None
        self._rows_written = 0

        # Start the writer thread
        self._start_writer_thread()

        logger.info(f"ResponseWriter initialized: {self.filepath}")

    def _start_writer_thread(self):
        """Start the background writer thread."""
        self._writer_thread = threading.Thread(target=self._write_loop, daemon=True)
        self._writer_thread.start()
        logger.debug("Background writer thread started")

    def _write_loop(self):
        """Background thread loop that writes responses to CSV."""
        try:
            while not self._stop_event.is_set():
                try:
                    # Get response from queue with timeout
                    response = self.response_queue.get(timeout=1.0)

                    # Write the response
                    self._write_response(response)

                    # Mark task as done
                    self.response_queue.task_done()

                except queue.Empty:
                    # Timeout - continue loop to check stop event
                    continue
                except Exception as e:
                    logger.error(f"Error in writer thread: {e}")

        except Exception as e:
            logger.error(f"Fatal error in writer thread: {e}")
        finally:
            # Close CSV file if open
            if self._csv_file:
                self._csv_file.close()
                logger.debug("CSV file closed")

    def _write_response(self, response: dict[str, Any]):
        """Write a single response to the CSV file.

        Parameters
        ----------
        response : Dict[str, Any]
            Response data to write
        """
        try:
            # Initialize CSV writer if not done yet
            if self._csv_writer is None:
                self._init_csv_writer(response)

            # Check if response has new fields not in current fieldnames
            assert self._fieldnames is not None, "Fieldnames must be initialized before writing."
            new_fields = set(response.keys()) - set(self._fieldnames)
            if new_fields:
                self._handle_new_fields(response, new_fields)

            # Write the response
            self._csv_writer.writerow(response)
            self._csv_file.flush()  # Ensure data is written to disk
            self._rows_written += 1

            if self._rows_written % 100 == 0:
                logger.debug(f"Written {self._rows_written} responses")

        except Exception as e:
            logger.error(f"Error writing response: {e}")

    def _init_csv_writer(self, first_response: dict[str, Any]):
        """Initialize the CSV writer with fieldnames from first response.

        Parameters
        ----------
        first_response : Dict[str, Any]
            First response to determine CSV fieldnames
        """
        self._fieldnames = list(first_response.keys())
        self._csv_file = open(self.filepath, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._fieldnames)
        self._csv_writer.writeheader()
        logger.info(f"CSV file initialized with fields: {self._fieldnames}")

    def _handle_new_fields(self, response: dict[str, Any], new_fields: set[str]):
        """Handle responses with new fields by recreating the CSV writer.

        Parameters
        ----------
        response : Dict[str, Any]
            Current response with new fields
        new_fields : set[str]
            Set of new field names not in current fieldnames
        """
        assert self._fieldnames is not None, "Fieldnames must be initialized before handling new fields."
        logger.warning(f"New fields detected: {new_fields}. Recreating CSV writer...")

        # Store current rows and close the file to release the lock
        if self._csv_file:
            self._csv_file.close()

        # Read all existing data from the CSV file
        existing_data = []
        if self.filepath.exists():
            try:
                with open(self.filepath, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    existing_data = list(reader)
            except FileNotFoundError:
                logger.warning("CSV file not found, creating a new one.")
            except Exception as e:
                logger.error(f"Error reading existing CSV data: {e}")

        # Update fieldnames with the new fields
        current_fieldnames = list(self._fieldnames)  # Copy existing fieldnames
        for field in new_fields:
            if field not in current_fieldnames:
                current_fieldnames.append(field)
        self._fieldnames = current_fieldnames

        # Recreate the CSV file with the updated fieldnames
        try:
            self._csv_file = open(self.filepath, "w", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._fieldnames)
            self._csv_writer.writeheader()

            # Write back the existing data
            for row in existing_data:
                self._csv_writer.writerow(row)

            logger.info(f"CSV file recreated with updated fields: {self._fieldnames}")

        except Exception as e:
            logger.error(f"Error recreating CSV file: {e}")

    def save_response(self, response: dict[str, Any]):
        """Save a response (non-blocking).

        Parameters
        ----------
        response : Dict[str, Any]
            Response data to save
        """
        if self._stop_event.is_set():
            logger.warning("Cannot save response - writer is stopped")
            return

        try:
            # Add timestamp if not present
            if "timestamp" not in response:
                response["timestamp"] = time.time()

            # Add to queue (non-blocking)
            self.response_queue.put(response, block=False)

        except queue.Full:
            logger.warning("Response queue is full - dropping response")
        except Exception as e:
            logger.error(f"Error queuing response: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get writer statistics.

        Returns
        -------
        Dict[str, Any]
            Writer statistics
        """
        return {
            "rows_written": self._rows_written,
            "queue_size": self.response_queue.qsize(),
            "is_alive": self._writer_thread.is_alive() if self._writer_thread else False,
            "filepath": str(self.filepath),
        }

    def close(self, timeout: float = 30.0):
        """Close the writer and wait for all responses to be written.

        Parameters
        ----------
        timeout : float, optional
            Maximum time to wait for queue to empty, by default 30.0
        """
        logger.info("Closing ResponseWriter...")

        # Wait for queue to empty
        start_time = time.time()
        while not self.response_queue.empty() and (time.time() - start_time) < timeout:
            time.sleep(0.1)

        if not self.response_queue.empty():
            logger.warning(f"Queue not empty after {timeout}s timeout - forcing close")

        # Stop the writer thread
        self._stop_event.set()

        # Wait for thread to finish
        if self._writer_thread and self._writer_thread.is_alive():
            self._writer_thread.join(timeout=5.0)
            if self._writer_thread.is_alive():
                logger.warning("Writer thread did not stop gracefully")

        logger.info(f"ResponseWriter closed. Total responses written: {self._rows_written}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
