import os
import csv
import threading
import queue
import atexit
from pathlib import Path
from typing import Optional
from loguru import logger
from . import COST_TYPE, COST_INFO, TEXT_PROCESSING
from .cost_handler import process_text, DB_SCHEMA


# Global state for the writer
_cost_queue: Optional[queue.Queue] = None
_writer_thread: Optional[threading.Thread] = None
_shutdown_event: Optional[threading.Event] = None
_csv_file_path: Optional[Path] = None
_is_initialized = False


def _worker_thread():
    """Background thread that processes the cost queue and writes to CSV."""
    logger.info(f"Cost writer thread started. Writing to: {_csv_file_path}")

    while not _shutdown_event.is_set() or not _cost_queue.empty():
        try:
            # Wait for items with timeout so we can check shutdown event
            cost_object = _cost_queue.get(timeout=0.5)

            try:
                _process_and_write_cost(cost_object)
            except Exception as e:
                logger.error(f"Error processing cost object: {e}", exc_info=True)
            finally:
                _cost_queue.task_done()

        except queue.Empty:
            continue

    logger.info("Cost writer thread shutting down")


def _process_and_write_cost(cost_object: dict):
    """Process a cost object and write records to CSV."""
    cost_type = cost_object.get(COST_TYPE)

    # Currently only TEXT_PROCESSING is implemented
    # You can add more handlers here for TTS, STT, etc.
    if cost_type == TEXT_PROCESSING or cost_type is None:
        records = list(process_text(cost_object, specified_type=cost_type))
    else:
        logger.warning(f"Unhandled cost type: {cost_type}. Defaulting to text processing.")
        records = list(process_text(cost_object, specified_type=cost_type))

    # Write records to CSV
    if records:
        _write_records_to_csv(records)


def _write_records_to_csv(records: list[dict]):
    """Append records to the CSV file."""
    try:
        # Check if file exists to determine if we need to write headers
        file_exists = _csv_file_path.exists()

        with open(_csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
            # Get field names from DB_SCHEMA
            fieldnames = list(DB_SCHEMA.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write header if file is new
            if not file_exists:
                writer.writeheader()
                logger.info(f"Created new CSV with headers: {fieldnames}")

            # Write all records
            for record in records:
                writer.writerow(record)

        logger.debug(f"Wrote {len(records)} cost record(s) to CSV")

    except Exception as e:
        logger.error(f"Failed to write to CSV: {e}", exc_info=True)


def initialize_cost_writer(experiment_id: Optional[str] = None):
    """
    Initialize the cost writer system.

    Args:
        experiment_id: The experiment ID. If None, will try to get from EXPERIMENT_ID env var.

    Raises:
        ValueError: If experiment_id is not provided and EXPERIMENT_ID env var is not set.
    """
    global _cost_queue, _writer_thread, _shutdown_event, _csv_file_path, _is_initialized

    if _is_initialized:
        logger.warning("Cost writer already initialized. Skipping.")
        return

    # Get experiment ID
    if experiment_id is None:
        experiment_id = os.environ.get('EXPERIMENT_ID')

    if not experiment_id:
        raise ValueError(
            "experiment_id must be provided or EXPERIMENT_ID environment variable must be set"
        )

    # Create experiments directory structure
    experiments_dir = Path("./experiments")
    experiment_dir = experiments_dir / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Set CSV file path
    _csv_file_path = experiment_dir / "experiment_cost_log.csv"

    # Initialize queue and shutdown event
    _cost_queue = queue.Queue()
    _shutdown_event = threading.Event()

    # Start worker thread
    _writer_thread = threading.Thread(target=_worker_thread, daemon=False)
    _writer_thread.start()

    _is_initialized = True

    # Register shutdown handler
    atexit.register(shutdown_cost_writer)

    logger.info(f"Cost writer initialized for experiment: {experiment_id}")
    logger.info(f"CSV output: {_csv_file_path.absolute()}")


def push_cost(cost_object: dict):
    """
    Push a cost object to the writer queue.

    Args:
        cost_object: Dictionary containing COST_TYPE, COST_INFO, and unique_id
    """
    if not _is_initialized:
        logger.error("Cost writer not initialized. Call initialize_cost_writer() first. Skipping cost logging.")
        return

    try:
        _cost_queue.put(cost_object, block=False)
        logger.debug(f"Pushed cost to queue. Queue size: {_cost_queue.qsize()}")
    except queue.Full:
        logger.error("Cost queue is full! Dropping cost record.")
    except Exception as e:
        logger.error(f"Failed to push cost to queue: {e}")


def shutdown_cost_writer(timeout: float = 5.0):
    """
    Gracefully shutdown the cost writer, waiting for queue to empty.

    Args:
        timeout: Maximum time to wait for queue to empty (in seconds)
    """
    global _is_initialized

    if not _is_initialized:
        return

    logger.info("Shutting down cost writer...")

    # Signal shutdown
    _shutdown_event.set()

    # Wait for queue to empty
    try:
        _cost_queue.join()
        logger.info("All cost records processed")
    except Exception as e:
        logger.warning(f"Error waiting for queue to empty: {e}")

    # Wait for thread to finish
    if _writer_thread and _writer_thread.is_alive():
        _writer_thread.join(timeout=timeout)
        if _writer_thread.is_alive():
            logger.warning(f"Writer thread did not finish within {timeout}s timeout")

    _is_initialized = False
    logger.info("Cost writer shutdown complete")


def get_csv_path() -> Optional[Path]:
    """Get the current CSV file path."""
    return _csv_file_path


def is_initialized() -> bool:
    """Check if the cost writer is initialized."""
    return _is_initialized