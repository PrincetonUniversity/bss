__version__ = '0.0.1'

import os.path
import logging.config
import yaml
import logging


def setup_logging(path='logging.yml'):
    path = os.path.join(os.path.dirname(__file__), path)
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO)


setup_logging()
logger = logging.getLogger('bss')
