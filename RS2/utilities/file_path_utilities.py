from multiprocessing import Pool

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *


def should_i_save_to_file(prediction: np.ndarray, results_list: List = None, export_pool: Pool = None):
    """
    There is a problem with python process communication that prevents us from communicating objects
    larger than 2 GB between processes (basically when the length of the pickle string that will be sent is
    communicated by the multiprocessing.Pipe object then the placeholder (I think) does not allow for long
    enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually
    patching system python code. We circumvent that problem here by saving the data to a npy file that will
    then be read (and finally deleted) by the background Process. The code running in the background process must be
    implemented such that it can take either filename (str) or np.ndarray as input

    This function determines whether the object that should be passed through a multiprocessing pipe is too big.

    It also determines whether the export pool can keep up with its tasks and if not it will trigger
    saving results to disk in order to reduce the amount of RAM that is consumed (queued tasks can use a lot of RAM)

    We also check for dead workers and crash in case there are any. This should fix some peoples issues where
    the inference was just stuck (due to out of memory problems).

    Returns: True if we should save to file else False
    """
    if prediction.nbytes * 0.85 > 2e9:  # *0.85 just to be safe
        print('INFO: Prediction is too large for python process-process communication. Saving to file...')
        return True
    if export_pool is not None:
        # check if we are still rockin' and rollin'
        check_is_pool_alive(export_pool)
        if results_list is not None:
            #    We should prevent the task queue from getting too long. This could cause lots of predictions being
            #    stuck in a queue and eating up memory. Best to save to disk instead in that case. Hopefully there
            #    will be fewer people with RAM issues in the future...
            if check_workers_busy(export_pool, results_list, allowed_num_queued=len(export_pool._pool)):
                return True
    return False


def check_is_pool_alive(export_pool: Pool):
    is_alive = [i.is_alive for i in export_pool._pool]
    if not all(is_alive):
        raise RuntimeError("Some workers in the export pool are no longer alive. That should not happen. You "
                           "probably don't have enough RAM :-(")


def check_workers_busy(export_pool: Pool, results_list: List, allowed_num_queued: int = 0):
    """

    returns True if the number of results that are not ready is greater than the number of available workers + allowed_num_queued
    """
    not_ready = [not i.ready() for i in results_list]
    if sum(not_ready) >= (len(export_pool._pool) + allowed_num_queued):
        return True
    return False

