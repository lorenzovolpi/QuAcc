import logging
import logging.handlers
import multiprocessing
import threading
from pathlib import Path
from typing import List

_logger_manager = None


class LoggerManager:
    def __init__(self, q, worker, listener=None, th=None):
        self.th: threading.Thread = th
        self.q: multiprocessing.Queue = q
        self.listener: logging.Logger = listener
        self._worker: List[logging.Logger] = [worker]
        self._listener_handlers: List[logging.Handler] = []

    def close(self):
        if self.th is not None:
            self.q.put(None)
            self.th.join()

    def rm_worker(self):
        self._worker.pop()

    @property
    def worker(self):
        return self._worker[-1]

    def new_worker(self):
        log = logging.getLogger(f"worker{len(self._worker)}")
        log.handlers.clear()
        self._worker.append(log)
        return log

    def add_listener_handler(self, rh):
        self._listener_handlers.append(rh)
        self.listener.addHandler(rh)
        self.listener.info("-" * 100)

    def clear_listener_handlers(self):
        for rh in self._listener_handlers:
            self.listener.removeHandler(rh)
        self._listener_handlers.clear()


def log_listener(root, q):
    while True:
        msg = q.get()
        if msg is None:
            return
        root.handle(msg)


def setup_logger():
    q = multiprocessing.Manager().Queue()

    log_file = "quacc.log"
    root_name = "listener"
    root = logging.getLogger(root_name)
    root.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    root.addHandler(fh)

    th = threading.Thread(target=log_listener, args=[root, q])
    th.start()

    worker_name = "worker"
    worker = logging.getLogger(worker_name)
    worker.setLevel(logging.DEBUG)
    qh = logging.handlers.QueueHandler(q)
    qh.setLevel(logging.DEBUG)
    qh.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s| %(levelname)-8s %(message)s",
            datefmt="%d/%m/%y %H:%M:%S",
        )
    )
    worker.addHandler(qh)

    global _logger_manager
    _logger_manager = LoggerManager(q, worker, listener=root, th=th)

    return _logger_manager.worker


def setup_worker_logger(q: multiprocessing.Queue = None):
    formatter = logging.Formatter(
        fmt="%(asctime)s| %(levelname)-12s%(message)s",
        datefmt="%d/%m/%y %H:%M:%S",
    )

    global _logger_manager
    if _logger_manager is None:
        worker_name = "worker"
        worker = logging.getLogger(worker_name)
        worker.setLevel(logging.DEBUG)
        qh = logging.handlers.QueueHandler(q)
        qh.setLevel(logging.DEBUG)
        qh.setFormatter(formatter)
        worker.addHandler(qh)

        _logger_manager = LoggerManager(q, worker)
        return _logger_manager.worker
    else:
        worker = _logger_manager.new_worker()
        worker.setLevel(logging.DEBUG)
        qh = logging.handlers.QueueHandler(_logger_manager.q)
        qh.setLevel(logging.DEBUG)
        qh.setFormatter(formatter)
        worker.addHandler(qh)
        return worker


def logger():
    return _logger_manager.worker


def logger_manager():
    return _logger_manager


def add_handler(path: Path):
    rh = logging.FileHandler(path, mode="a")
    rh.setLevel(logging.DEBUG)
    _logger_manager.add_listener_handler(rh)


def clear_handlers():
    _logger_manager.clear_listener_handlers()
