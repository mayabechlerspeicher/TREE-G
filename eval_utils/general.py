import os
import logging
import time
import pickle
import json
from sklearn.decomposition import PCA
from timeit import default_timer as timer
from datetime import timedelta

class Timer():
    def __init__(self):
        self.start = timer()

    def end(self):
        end = timer()
        return timedelta(seconds=end - self.start)


class LogFormatter():
    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime('%x %X'),
            timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message)


def create_logger(log_dir, log_name='', dump=True):
    if len(log_name):
        filepath = os.path.join(log_dir, '{}.log'.format(log_name))
    else:
        filepath = os.path.join(log_dir, 'log.log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # # Safety check
    # if os.path.exists(filepath) and opt.checkpoint == "":
    #     logging.warning("Experiment already exists!")

    # Create logger
    log_formatter = LogFormatter()

    if dump:
        # create file handler and set level to info
        file_handler = logging.FileHandler(filepath, "a")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to info
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if dump:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()
    logger.reset_time = reset_time

    logger.info('Created main log at ' + str(filepath))
    return logger


def load_pickle(pkl_filename):
    with open(pkl_filename, 'rb') as f:
        x = pickle.load(f)
    return x


def save_pickle(x, pkl_filename):
    with open(pkl_filename, 'wb') as file:
        pickle.dump(x, file)


def save_json(x, json_filename):
    with open(json_filename, 'w') as f:
        json.dump(x, f)


def load_json(json_filename):
    with open(json_filename, 'r') as f:
        x = json.load(f)
    return x


def reduce_dim(X, out_dim, verbose=False):
    pca = PCA(out_dim)
    X_reduced = pca.fit_transform(X)
    if verbose:
        logging.info('Applied PCA to {} dim. PCA variance: {:.4f}'.format(out_dim, sum(pca.explained_variance_ratio_)))
    return X_reduced, pca
