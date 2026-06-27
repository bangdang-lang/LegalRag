import logging
import inspect

def get_logger(name=None):
    """Return a configured logger for the calling module or provided name."""
    if name is None:
        # infer caller module name
        frame = inspect.stack()[1]
        module = inspect.getmodule(frame[0])
        name = module.__name__ if module else '__main__'
    logger = logging.getLogger(name)
    # Avoid adding multiple handlers if already configured elsewhere
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(name)s:%(lineno)d: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger
