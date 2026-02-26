import queue
import threading
import uuid
from typing import Any, Callable

class JobQueue:
    def __init__(self):
        self._queue = queue.Queue()
        self._results = {}
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._process_jobs, daemon=True)
        self._worker.start()

    def _process_jobs(self):
        while True:
            try:
                item = self._queue.get()
                if item is None:
                    break
                job_id, fn, args, kwargs = item
                try:
                    result = fn(*args, **kwargs)
                except Exception as e:
                    result = e
                with self._lock:
                    self._results[job_id] = result
            finally:
                self._queue.task_done()

    def add_job(self, fn: Callable, *args, **kwargs) -> str:
        job_id = str(uuid.uuid4())
        self._queue.put((job_id, fn, args, kwargs))
        return job_id

    def get_result(self, job_id: str) -> Any:
        with self._lock:
            return self._results.get(job_id, None)

    def shutdown(self):
        self._queue.put(None)
        self._worker.join()

if __name__ == "__main__":
    import time

    def sample_task(x, y):
        time.sleep(1)
        return x + y

    jq = JobQueue()
    id1 = jq.add_job(sample_task, 1, 2)
    id2 = jq.add_job(sample_task, 3, 4)

    time.sleep(2)

    print(f"Result 1: {jq.get_result(id1)}")
    print(f"Result 2: {jq.get_result(id2)}")

    jq.shutdown()
