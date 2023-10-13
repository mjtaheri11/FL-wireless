import logging
from datetime import datetime

from jsonformatter import JsonFormatter  # type: ignore

LOGGER_NAME = "FL-wireless"


class LoggingLevel:
    CRITICAL = logging.CRITICAL
    ERROR = logging.ERROR
    WARNING = logging.WARNING
    INFO = logging.INFO
    DEBUG = logging.DEBUG
    

def get_logger():
    # create a custom logger
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:  # logger is already setup, don't setup again
        return logger
    logger.propagate = False
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler("./FL-wireless.log", encoding="utf8")
    stream_handler = logging.StreamHandler()
    formatter = JsonFormatter(
        ensure_ascii=False,
        mix_extra=True,
        mix_extra_position="head",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def simple_logger(message, log_level=LoggingLevel.INFO):
    logger = get_logger()
    logger.log(
        log_level,
        message,
        extra={
            "logtime": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        },
    )
    