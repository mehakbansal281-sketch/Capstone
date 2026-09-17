import os
import logging
import pytest
from src.ingestion import DataIngestionPipeline

def test_log_file_creation(tmp_path):
    """
    Verifies that logging creates a log file in the designated test log directory.
    """
    test_log_dir = tmp_path / "test_logs"
    test_log_dir.mkdir()
    test_log_file = test_log_dir / "test_app.log"

    test_logger = logging.getLogger("TestLogger")
    test_logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(str(test_log_file))
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    test_logger.addHandler(file_handler)

    test_logger.info("Test log entry: Application started successfully.")

    file_handler.close()
    test_logger.removeHandler(file_handler)

    assert test_log_file.exists()
    content = test_log_file.read_text()
    assert "Test log entry: Application started successfully." in content

def test_log_format_validation(tmp_path):
    """
    Verifies that the log messages conform to the expected format (Timestamp - Name - Level - Message).
    """
    test_log_file = tmp_path / "formatted.log"
    logger = logging.getLogger("FormatLogger")
    logger.setLevel(logging.DEBUG)

    handler = logging.FileHandler(str(test_log_file))
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)

    logger.debug("Debug event logged")
    logger.info("Info event logged")
    logger.warning("Warning event logged")

    handler.close()
    logger.removeHandler(handler)

    lines = test_log_file.read_text().strip().split("\n")
    assert len(lines) == 3
    for line in lines:
        parts = line.split(" - ")
        assert len(parts) >= 4
        assert parts[1] == "FormatLogger"
        assert parts[2] in ["DEBUG", "INFO", "WARNING"]

def test_error_logging(tmp_path):
    """
    Verifies that errors and exceptions are logged properly with tracebacks.
    """
    test_log_file = tmp_path / "error.log"
    logger = logging.getLogger("ErrorLogger")
    logger.setLevel(logging.ERROR)

    handler = logging.FileHandler(str(test_log_file))
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)

    try:
        raise ValueError("Simulated data validation error")
    except Exception as exc:
        logger.error("Data pipeline failed", exc_info=True)

    handler.close()
    logger.removeHandler(handler)

    content = test_log_file.read_text()
    assert "ERROR" in content
    assert "Data pipeline failed" in content
    assert "Simulated data validation error" in content
    assert "Traceback" in content
