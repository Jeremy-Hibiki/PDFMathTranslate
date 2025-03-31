from threading import RLock
from typing import Any


class SingletonMeta(type):
    _instances: dict[type, Any] = {}
    _lock: RLock = RLock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
