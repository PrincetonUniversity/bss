from concurrent import futures
import numpy as np

from config import settings
from utils import Benchmark
from real_eval import run_test

if __name__ == '__main__':

    executor = futures.ProcessPoolExecutor(max_workers=settings['real_eval']['workers'])

    filePrefixes = [p.strip() for p in settings['real_eval']['filePrefix'].split(',')]
    n = len(filePrefixes)
    results = executor.map(
        run_test,
        [settings['real_eval']['directory']]*n,
        filePrefixes,
        [settings['real_eval']['dims']]*n,
        [settings['real_eval']['burnin']]*n,
        [settings['real_eval']['iterations']]*n,
        [settings['real_eval']['np.seed']]*n
    )

    with Benchmark('all tests'):
        for i, result in enumerate(results):
            pass