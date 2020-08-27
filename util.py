"""
公用代码
"""

import os

from datetime import datetime

LTP_DIR = "/data/private/zms/model_weight/ltp4-base/"
SKEP_DIR = "./ernie_skep_sentiment_analysis/"


def output(*args):
    message = ''.join([str(arg) for arg in args])
    print(f"[{str(datetime.now())[:-7]} - {os.getpid()}] {message}")


def generate_batch(data, batch_size):
    iterable = range(0, len(data) - batch_size, batch_size)
    for i in iterable:
        yield data[i: i + batch_size]
    else:
        yield data[i + batch_size: len(data)]
