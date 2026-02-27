import threading
import queue
import uuid
from typing import Any, Callable, Optional

class JobQueue:
    def __init__(self):
        self._queue = queue.Queue()
        self._results = {}
        self._lock = threading.Lock()
        self._running = True
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def add_job(self, fn: Callable, *args, **kwargs) -> str:
        job_id = str(uuid.uuid4())
        with self._lock:
            self._results[job_id] = {'result': None, 'done': False}
        self._queue.put((fn, args, kwargs, job_id))
        return job_id

    def get_result(self, job_id: str) -> Optional[Any]:
        with self._lock:
            if job_id not in self._results:
                return None
            job_info = self._results[job_id]
            if job_info['done']:
                return job_info['result']
            return None

    def _worker_loop(self):
        while self._running:
            try:
                fn, args, kwargs, job_id = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            try:
                result = fn(*args, **kwargs)
                with self._lock:
                    self._results[job_id]['result'] = result
                    self._results[job_id]['done'] = True
            except Exception as e:
                with self._lock:
                    self._results[job_id]['result'] = e
                    self._results[job_id]['done'] = True
            
            self._queue.task_done()

    def shutdown(self):
        self._running = False
        self._worker.join()
