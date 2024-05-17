import multiprocessing
from functools import partial
import logging
import pytorch_common.util as pu


class MapFn:
    def __init__(self, fn, fallback_result=None):
        self.fn              = fn
        self.fallback_result = fallback_result

    def __call__(self, params):
        try:
            return self.fn(*params)
        except Exception as err:
            logging.error(f'Error to perform function with args: {list(params)}. Cause: {err}')
            return self.fallback_result


class ParallelExecutor:
    def __init__(self, n_processes = 10):
        self.__pool = multiprocessing.Pool(processes=n_processes)

    def __call__(self, fn, params, fallback_result=None):
        sw = pu.Stopwatch()
        fn = MapFn(fn, fallback_result)
        result = self.__pool.map(fn, params)
        logging.info(sw.to_str())
        return result