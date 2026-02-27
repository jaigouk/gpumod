import threading
import time
import uuid
from collections import deque
from typing import Any, Callable

class JobQueue:
    def __init__(self):
        self.jobs = deque()
        self.results = {}
        self.lock = threading.Lock()
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()

    def add_job(self, fn: Callable, *args, priority: int = 5, **kwargs) -> str:
        job_id = str(uuid.uuid4())
        with self.lock:
            self.jobs.append((priority, time.time(), job_id, fn, args, kwargs))
            self.jobs = deque(sorted(self.jobs, key=lambda x: (-x[0], x[1])))
        return job_id

    def get_result(self, job_id: str) -> Any:
        with self.lock:
            return self.results.get(job_id)

    def _worker(self):
        while True:
            job = None
            with self.lock:
                if self.jobs:
                    job = self.jobs.popleft()

            if job:
                priority, ts, job_id, fn, args, kwargs = job
                try:
                    result = self._execute_with_retry(fn, args, kwargs)
                    with self.lock:
                        self.results[job_id] = result
                except Exception as e:
                    with self.lock:
                        self.results[job_id] = None
            else:
                time.sleep(0.01)

    def _execute_with_retry(self, fn, args, kwargs, max_retries=3):
        backoff = 1
        for attempt in range(max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception:
                if attempt == max_retries:
                    raise
                time.sleep(backoff)
                backoff *= 2
