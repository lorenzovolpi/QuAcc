import logging
import logging.handlers
import multiprocessing
import threading
from pathlib import Path
from typing import List


class Logger:
    __logger_file = "quacc.log"
    __logger_name = "queue_logger"
    __manager = None
    __queue = None
    __thread = None
    __setup = False
    __handlers = []

    @classmethod
    def __logger_listener(cls, q):
        while True:
            record = q.get()
            if record is None:
                break
            root = logging.getLogger("listener")
            root.handle(record)

    @classmethod
    def setup(cls):
        if cls.__setup:
            return

        # setup root
        root = logging.getLogger("listener")
        root.setLevel(logging.DEBUG)
        rh = logging.FileHandler(cls.__logger_file, mode="a")
        rh.setLevel(logging.DEBUG)
        root.addHandler(rh)

        # setup logger
        if cls.__manager is None:
            cls.__manager = multiprocessing.Manager()

        if cls.__queue is None:
            cls.__queue = cls.__manager.Queue()

        logger = logging.getLogger(cls.__logger_name)
        logger.setLevel(logging.DEBUG)
        qh = logging.handlers.QueueHandler(cls.__queue)
        qh.setLevel(logging.DEBUG)
        qh.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s| %(levelname)-8s %(message)s",
                datefmt="%d/%m/%y %H:%M:%S",
            )
        )
        logger.addHandler(qh)

        # start listener
        cls.__thread = threading.Thread(
            target=cls.__logger_listener,
            args=(cls.__queue,),
        )
        cls.__thread.start()

        cls.__setup = True

    @classmethod
    def add_handler(cls, path: Path):
        root = logging.getLogger("listener")
        rh = logging.FileHandler(path, mode="a")
        rh.setLevel(logging.DEBUG)
        cls.__handlers.append(rh)
        root.addHandler(rh)
        root.info("-" * 100)

    @classmethod
    def clear_handlers(cls):
        root = logging.getLogger("listener")
        for h in cls.__handlers:
            root.removeHandler(h)
        cls.__handlers.clear()

    @classmethod
    def queue(cls):
        if not cls.__setup:
            cls.setup()

        return cls.__queue

    @classmethod
    def logger(cls):
        if not cls.__setup:
            cls.setup()

        return logging.getLogger(cls.__logger_name)

    @classmethod
    def close(cls):
        if cls.__setup and cls.__thread is not None:
            cls.__queue.put(None)
            cls.__thread.join()
            # cls.__manager.close()


class SubLogger:
    __queue = None
    __setup = False

    @classmethod
    def setup(cls, q):
        if cls.__setup:
            return

        cls.__queue = q

        # setup root
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)
        rh = logging.handlers.QueueHandler(q)
        rh.setLevel(logging.DEBUG)
        rh.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s| %(levelname)-12s%(message)s",
                datefmt="%d/%m/%y %H:%M:%S",
            )
        )
        root.addHandler(rh)

        cls.__setup = True

    @classmethod
    def logger(cls):
        if not cls.__setup:
            return None

        return logging.getLogger()


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
