import os as _os
import logging


DIR = _os.path.dirname(_os.path.abspath(__file__))
DATA_DIR = _os.path.join(DIR, 'data')

# https://docs.python.org/3/howto/logging.html
logging.getLogger(__name__).addHandler(logging.NullHandler())


def set_basic_logging_config():
    """
    Use this function to set up a basic streaming log config.
    """
    from logging.config import dictConfig
    logging_config = dict(
        version=1,
        formatters={
            'f': {'format':
                      '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'}
        },
        handlers={
            'h': {'class': 'logging.StreamHandler',
                  'formatter': 'f',
                  'level': logging.INFO}
        },
        root={
            'handlers': ['h'],
            'level': logging.INFO,
        },
    )
    dictConfig(logging_config)
