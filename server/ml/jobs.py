"""One background training job at a time, with event fan-out for the SSE stream."""
from __future__ import annotations

import queue
import threading

from . import train


class Job:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._subs: list[queue.Queue] = []
        self.run_id: int | None = None

    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def emit(self, event: dict) -> None:
        for q in list(self._subs):
            q.put(event)

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        if q in self._subs:
            self._subs.remove(q)

    def start(self, params: dict | None = None, resume_id: int | None = None) -> int | None:
        """Start a new run, or resume `resume_id`. None when a run is already going."""
        with self._lock:
            if self.running():
                return None
            run_id = resume_id if resume_id is not None else train.create_run(params or {})
            self.run_id = run_id
            self._stop.clear()
            self._thread = threading.Thread(target=train.train, name="train", daemon=True,
                                            args=(run_id, self.emit, self._stop, resume_id is not None))
            self._thread.start()
            return run_id

    def stop(self) -> bool:
        if not self.running():
            return False
        self._stop.set()
        return True


job = Job()
