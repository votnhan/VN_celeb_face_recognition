import os
import logging
from .logger import setup_logging
from datetime import datetime


def get_logger_for_run(log_container):
    run_id = datetime.now().strftime(r'%m%d_%H%M%S')
    log_dir = os.path.join(log_container, run_id)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    setup_logging(log_dir)
    logger = logging.getLogger(os.environ['LOGGER_ID'])
    return logger, log_dir
