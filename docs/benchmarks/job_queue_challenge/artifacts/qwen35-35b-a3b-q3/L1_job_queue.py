import threading
import queue
import uuid
from typing import Any, Callable, Optional

class JobQueue:
    def __init__(self, num_workers: int = 4):
        self._queue = queue.Queue()
        self._results = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._workers = []

        for _ in range(num_workers):
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()
            self._workers.append(t)

    def add_job(self, fn: Callable, *args, **kwargs) -> str:
        job_id = str(uuid.uuid4())
        self._queue.put((job_id, fn, args, kwargs))
        return job_id

    def get_result(self, job_id: str) -> Optional[Any]:
        with self._lock:
            return self._results.get(job_id)

    def _worker(self):
        while not self._stop_event.is_set():
            try:
                job_id, fn, args, kwargs = self._queue.get(timeout=1)
                try:
                    result = fn(*args, **kwargs)
                    with self._lock:
                        self._results[job_id] = result
                except Exception as e:
                    with self._lock:
                        self._results[job_id] = e
                finally:
                    self._queue.task_done()
            except queue.Empty:
                continue

    def shutdown(self):
        self._stop_event.set()
        for t in self._workers:
            t.join()
