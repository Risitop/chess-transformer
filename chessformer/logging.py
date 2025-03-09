"""A minimal common custom logging module to avoid using the root logger. This
module centralizes the logging of all the modules of the project. It is build as
a thin wrapper around the standard logging module.

Log files are stored in the `.logs/` directory, located of the ScientaLab directory
root. The log files are stored in directories named `logs-YYYY-MM-DD-HHh` where
`YYYY-MM-DD-HHh` is the current date and hour.

Basic usage
------------
>>> from common import logging
>>> logging.debug("This is a debug message.") # Log file only
>>> logging.info("This is an info message.") # Log file and console
>>> logging.warning("This is a warning message.") # Log file and console
>>> logging.error("This is an error message.") # Log file and console
>>> logging.critical("This is a critical message.") # Log file and console

Changing the logging level
---------------------------
>>> from common import logging
>>> logging.set_level(logging.DEBUG) # Set the console logging level to DEUBG
>>> logging.debug("This is a debug message.") # Will be displayed in the console

Happy logging!
"""

import logging
import os
import re
import shutil
import sys
import traceback
import warnings
from datetime import datetime
from pathlib import Path

import pygit2

LOG_DIR = Path(__file__).parent.parent / ".logs"
_FORBIDDEN_FILE_CHARS = '{}[]()/<>:"/\\|?*. '
_LOGGER = None
_LOG_FILE = None

# Forwarding log levels
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL


class _TermColorFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    cyan = "\x1b[36;20m"
    reset = "\x1b[0m"

    format_time = cyan + "{asctime}" + reset
    format_file = "#file#"
    format_msg = "{message}"

    FORMATS = {
        logging.DEBUG: f"({format_time}) {format_file} > {grey}#DEBUG# {format_msg}{reset}",
        logging.INFO: f"({format_time}) {format_msg}",
        logging.WARNING: f"({format_time}) {yellow}{format_file} > {format_msg}{reset}",
        logging.ERROR: f"({format_time}) {bold_red}{format_file} > {format_msg}{reset}",
        logging.CRITICAL: f"({format_time}) {bold_red}{format_file} > {format_msg}{reset}",
    }

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, style="{", datefmt="%H:%M:%S")
        formatted = formatter.format(record)
        if record.levelno != INFO:
            formatted = formatted.replace(self.format_file, _get_caller())
        else:
            formatted = formatted.replace(self.format_file, "")
        return formatted


class _FileFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_fmt = "[{asctime}] {levelname} #file# > {message}"
        formatter = logging.Formatter(log_fmt, style="{")
        formatted = formatter.format(record).replace("#file#", _get_caller())
        return formatted


def _create_logger():
    if _LOGGER is not None:
        warning("Logger already created, please report this issue.")
        return _LOGGER

    # Set root logger to WARNING to avoid double logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # create a folder for the hour if it doesn't exist
    log_dir = LOG_DIR / datetime.now().strftime("logs-%Y-%m-%d-%Hh")
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir)
        except FileExistsError as _:  # Avoids multi-process racing condition
            pass
    fname = f"log_{datetime.now()}"
    for c in _FORBIDDEN_FILE_CHARS:
        fname = fname.replace(c, "_")
    global _LOG_FILE
    _LOG_FILE = log_dir / f"{fname}.log"
    file_handler = logging.FileHandler(_LOG_FILE)
    formatter = _FileFormatter()
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Create a stream handler for the console
    stream_handler = logging.StreamHandler()
    stream_handler.stream = sys.__stdout__  # type: ignore
    stream_handler.setLevel(logging.INFO)
    formatter = _TermColorFormatter()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    sys.excepthook = _handle_exception

    return logger


def _handle_exception(exc_type, exc_value, exc_traceback):
    """Route uncaught exceptions to the logger."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    critical(f"Uncaught exception: {exc_type.__name__}: {exc_value}")
    critical("".join(traceback.format_exception(exc_type, exc_value, exc_traceback)))


def _log_msg(msg: str, level: int = logging.DEBUG) -> None:
    """Base logging function with formatting features."""
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = _create_logger()
        _greet()
    for line in msg.strip().split("\n"):
        try:  # Avoids logging errors to crash the program :o)
            _LOGGER.log(level, line)
        except Exception as _:
            pass


def debug(msg: str) -> None:
    """Print a debug message."""
    _log_msg(msg, logging.DEBUG)


def info(msg: str) -> None:
    """Print an information message."""
    _log_msg(msg, logging.INFO)


def warning(msg: str, raise_warning: bool = True) -> None:
    """Print a warning message."""
    _log_msg(msg, logging.WARNING)
    if raise_warning:
        warnings.warn(msg)


def error(msg: str) -> None:
    """Print an error message. Does not raise an exception."""
    _log_msg(msg, logging.ERROR)


def critical(msg: str) -> None:
    """Print a critical message. Does not raise an exception."""
    _log_msg(msg, logging.CRITICAL)


def set_level(level: int) -> None:
    """Set the terminal logging level, use the LogLevels class."""
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = _create_logger()
        _greet()
    _LOGGER.setLevel(level)
    for handler in _LOGGER.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(level)


def _get_caller() -> str:
    """Get the file and line of the caller."""
    try:
        # Trick to find the caller of the logger
        index = 0
        for i, s in enumerate(traceback.format_stack()):
            if "logging.py" in s:
                index = i
                break
        prev = traceback.format_stack()[index - 1]
        file = re.search('File "(.*?)"', prev).group(1).split("/")[-1]  # type: ignore
        line = re.search("line (.*?),", prev).group(1)  # type: ignore
        return f"{file}:{line}"
    except IndexError:
        return "(unknown)"


def basicConfig(*args, **kwargs) -> None:
    """Does nothing, will raise a warning as basicConfig is discouraged."""
    warning(
        "-- The basicConfig function is discouraged, use logging.set_level instead.\n"
        "-- This is discouraged because it would affect the root logger."
    )


def copy_log_file(dest_path: Path) -> None:
    """Copy the current session log file to the specified directory.

    Parameters
    ----------
    dest_path : Path
        Path to the directory where the log file will be copied.
    """
    if _LOGGER is None or _LOG_FILE is None:
        warning("No log file found to copy.")
        return
    if not os.path.exists(_LOG_FILE):
        warning(f"Log file at {_LOG_FILE} does not exist.")
        return
    shutil.copyfile(_LOG_FILE, dest_path)


def delete_log_file() -> None:
    """Delete the current session log file."""
    if _LOGGER is None or _LOG_FILE is None:
        return
    if not os.path.exists(_LOG_FILE):
        return
    os.remove(_LOG_FILE)


def log_file_exists() -> bool:
    """Check if the current session log file exists."""
    if _LOGGER is None or _LOG_FILE is None:
        return False
    return os.path.exists(_LOG_FILE)


def _greet():
    """Print some info about the environment."""
    debug("Logger initialized.")
    debug(f"Log files are stored in the directory: {LOG_DIR}")
    debug(f"System OS: {os.name}")
    debug(f"System platform: {sys.platform}")
    debug(f"System version: {sys.version}")
    try:
        repo = pygit2.Repository(".")  # type: ignore
        debug(f"Git repository found at {repo.path}")
        debug(f"Current branch: {repo.head.shorthand}")
        debug(f"Current commit: {repo.head.target}")
    except Exception as e:
        debug(f"Git repository not found: {e}")
    try:
        with os.popen("pip freeze") as f:
            debug("$ pip freeze")
            for line in f:
                debug(line.strip())
    except Exception as e:
        debug(f"Failed to list installed packages: {e}")
    debug("End of logger initialization.")
