import logging
import traceback

import colorama
from colorama import Fore, Back, Style
colorama.init()


class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors
    reference: https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output/
    """

    import platform
    if platform.system() == 'Windows':
        grey = "\x1b[38;21m"
        yellow = "\x1b[33;21m"
        magenta = "\x1b[35;21m"
        red = "\x1b[31;21m"
        reset = "\x1b[0m"
    else:
        grey = Style.DIM
        yellow = Fore.YELLOW
        magenta = Fore.MAGENTA
        red = Fore.RED
        reset = Style.RESET_ALL
    # format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class Logger:

    def __init__(self, name=None):
        # create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # create console handler and set level to debug
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        # add formatter
        handler.setFormatter(CustomFormatter())

        # add handler to logger
        self.logger.addHandler(handler)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warn(self, message):
        self.logger.warning(message)

    def print_stack(self):
        print("---traceback---")
        for line in traceback.format_stack():
            print(line.strip())

    def error(self, message):
        self.logger.error(message)
        self.print_stack()
        exit()

    def critical(self, message):
        self.logger.critical(message)
        self.print_stack()
