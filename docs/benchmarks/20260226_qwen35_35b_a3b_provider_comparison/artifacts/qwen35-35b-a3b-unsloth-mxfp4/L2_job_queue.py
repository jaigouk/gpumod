import threading
import queue
import uuid
import time
from typing import Any, Callable, Dict, Optional

class JobQueue:
    def __init__(self):
        self._queue = queue.Queue()
        self._results: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def _worker_loop(self):
        while True:
            job = self._queue.get()
            if job is None:
                break
            job_id = job['job_id']
            fn = job['fn']
            args = job['args']
            kwargs = job['kwargs']
            
            max_attempts = 4
            backoff_times = [1, 2, 4]
            attempts = 0
            
            while attempts < max_attempts:
                try:
                    result = fn(*args, **kwargs)
                    with self._lock:
                        self._results[job_id] = result
                    break
                except Exception as e:
                    attempts += 1
                    if attempts < max_attempts:
                        time.sleep(backoff_times[attempts - 1])
                    else:
                        with self._lock:
                            self._results[job_id] = e
            
            self._queue.task_done()

    def add_job(self, fn: Callable, *args, **kwargs) -> str:
        job_id = str(uuid.uuid4())
        job = {
            'job_id': job_id,
            'fn': fn,
            'args': args,
            'kwargs': kwargs
        }
        self._queue.put(job)
        return job_id

    def get_result(self, job_id: str) -> Optional[Any]:
        with self._lock:
            res = self._results.get(job_id)
            if res is None:
                return None
            if isinstance(res, Exception):
                raise res
            return res
