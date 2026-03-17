"""Conversation logging utility for storing full prompt-response pairs."""

import csv
import queue
import threading
import time
from pathlib import Path

from beam_abm.common.logging import get_logger

logger = get_logger(__name__)


class ConversationLogger:
    """Logger for storing full LLM conversations (prompts and responses)."""

    def __init__(self, output_dir: str, filename: str = "conversations.csv"):
        """Initialize the conversation logger.

        Parameters
        ----------
        output_dir : str
            Directory to save the conversation file
        filename : str, optional
            Name of the CSV file, by default "conversations.csv"
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = self.output_dir / filename

        # Queue for conversations to be written
        self.conversation_queue = queue.Queue()

        # Threading control
        self._stop_event = threading.Event()
        self._writer_thread = None
        self._csv_file = None
        self._csv_writer = None
        self._fieldnames = [
            "timestamp",
            "record_index",
            "record_id",
            "prompt_strategy",
            "perturbation",
            "model_name",
            "method",
            "prompt_text",
            "response_text",
            "thinking_content",  # New field for reasoning/thinking
            "reasoning_effort",
            "reasoning_tokens",
            "reasoning_summary",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "validation_success",
            "validation_error",
            "attempt_number",
            "request_id",
            # Basic performance metrics
            "inference_time_seconds",
            "tokens_per_second",
            "optimization_applied",
            "estimated_throughput",
            "max_num_seqs",
            "max_num_batched_tokens",
            "gpu_memory_utilization",
        ]
        self._rows_written = 0

        # Start the writer thread
        self._start_writer_thread()

        logger.info(f"ConversationLogger initialized: {self.filepath}")

    def _start_writer_thread(self):
        """Start the background writer thread."""
        self._writer_thread = threading.Thread(target=self._write_loop, daemon=True)
        self._writer_thread.start()

    def _write_loop(self):
        """Background thread for writing conversations to CSV."""
        try:
            # Open CSV file for writing
            self._csv_file = open(self.filepath, "w", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._fieldnames)
            self._csv_writer.writeheader()
            self._csv_file.flush()

            while not self._stop_event.is_set():
                try:
                    # Get conversation from queue with timeout
                    conversation = self.conversation_queue.get(timeout=0.1)

                    if conversation is None:  # Sentinel value to stop
                        break

                    # Write the conversation
                    self._csv_writer.writerow(conversation)
                    self._csv_file.flush()
                    self._rows_written += 1

                    # Mark task as done
                    self.conversation_queue.task_done()

                except queue.Empty:
                    continue
                except (OSError, csv.Error, ValueError) as e:
                    logger.error(f"Error writing conversation: {e}")

        except (OSError, csv.Error, ValueError) as e:
            logger.error(f"Error in conversation writer thread: {e}")
        finally:
            if self._csv_file:
                self._csv_file.close()

    def log_conversation(
        self,
        record_index: int,
        model_name: str,
        method: str,
        prompt_text: str,
        response_text: str,
        record_id: str | None = None,
        prompt_strategy: str | None = None,
        perturbation: str | None = None,
        thinking_content: str = "",
        reasoning_effort: str = "",
        reasoning_tokens: int | None = None,
        reasoning_summary: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
        validation_success: bool = True,
        validation_error: str | None = None,
        attempt_number: int = 1,
        request_id: str | None = None,
        # Basic performance metrics
        inference_time_seconds: float = 0.0,
        tokens_per_second: float = 0.0,
        optimization_applied: bool = False,
        estimated_throughput: float = 0.0,
        max_num_seqs: int = 0,
        max_num_batched_tokens: int = 0,
        gpu_memory_utilization: float = 0.0,
    ):
        """Log a conversation to the CSV file.

        Parameters
        ----------
        record_index : int
            Index of the record being processed
        model_name : str
            Name of the LLM model used
        method : str
            Processing method (e.g., "Zero_Shot", "Few_Shot")
        prompt_text : str
            The full prompt sent to the model
        response_text : str
            The full response received from the model
        input_tokens : int, optional
            Number of input tokens, by default 0
        output_tokens : int, optional
            Number of output tokens, by default 0
        total_tokens : int, optional
            Total tokens used, by default 0
        validation_success : bool, optional
            Whether response validation succeeded, by default True
        validation_error : str, optional
            Error message if validation failed, by default None
        attempt_number : int, optional
            Attempt number for this request, by default 1
        request_id : str, optional
            Unique request identifier, by default None
        """
        conversation = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "record_index": record_index,
            "record_id": "" if record_id is None else record_id,
            "prompt_strategy": "" if prompt_strategy is None else prompt_strategy,
            "perturbation": "" if perturbation is None else perturbation,
            "model_name": model_name,
            "method": method,
            "prompt_text": prompt_text,
            "response_text": response_text,
            "thinking_content": thinking_content,
            "reasoning_effort": reasoning_effort,
            "reasoning_tokens": "" if reasoning_tokens is None else reasoning_tokens,
            "reasoning_summary": reasoning_summary,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "validation_success": validation_success,
            "validation_error": validation_error or "",
            "attempt_number": attempt_number,
            "request_id": request_id or "",
            # Basic performance metrics
            "inference_time_seconds": inference_time_seconds,
            "tokens_per_second": tokens_per_second,
            "optimization_applied": optimization_applied,
            "estimated_throughput": estimated_throughput,
            "max_num_seqs": max_num_seqs,
            "max_num_batched_tokens": max_num_batched_tokens,
            "gpu_memory_utilization": gpu_memory_utilization,
        }

        try:
            self.conversation_queue.put(conversation, timeout=1.0)
        except queue.Full:
            logger.warning("Conversation queue full, dropping conversation log")

    def close(self):
        """Close the conversation logger and clean up resources."""
        if self._writer_thread and self._writer_thread.is_alive():
            # Signal the thread to stop
            self._stop_event.set()

            # Send sentinel value to stop the writer
            try:
                self.conversation_queue.put(None, timeout=1.0)
            except queue.Full:
                pass

            # Wait for the thread to finish
            self._writer_thread.join(timeout=2.0)

            logger.info(f"ConversationLogger closed. Total conversations logged: {self._rows_written}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
