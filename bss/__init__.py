"""
The implementations are based on the following publication:

.. [Bansal2005] "Bayesian Structured Sparsity from Gaussian Fields",
   Barbara E. Engelhardt, Ryan P. Adams
   https://arxiv.org/abs/1407.2235
"""

__version__ = '0.0.2'

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
