import queue
import threading
import uuid
import time
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
                self._retry_job(job_id, fn, args, kwargs)
            finally:
                self._queue.task_done()

    def _retry_job(self, job_id: str, fn: Callable, args: tuple, kwargs: dict):
        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                result = fn(*args, **kwargs)
                with self._lock:
                    self._results[job_id] = result
                return
            except Exception as e:
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                else:
                    with self._lock:
                        self._results[job_id] = e

    def add_job(self, fn: Callable, *args, **kwargs) -> str:
        job_id = str(uuid.uuid4())
        self._queue.put((job_id, fn, args, kwargs))
        return job_id

    def get_result(self, job_id: str) -> Any:
        with self._lock:
            result = self._results.get(job_id, None)
            if isinstance(result, Exception):
                raise result
            return result

    def shutdown(self):
        self._queue.put(None)
        self._worker.join()

if __name__ == "__main__":
    import time

    def sample_task(x, y):
        time.sleep(1)
        return x + y

    def failing_task():
        raise ValueError("Task failed")

    jq = JobQueue()
    id1 = jq.add_job(sample_task, 1, 2)
    id2 = jq.add_job(sample_task, 3, 4)
    id3 = jq.add_job(failing_task)

    time.sleep(10)

    print(f"Result 1: {jq.get_result(id1)}")
    print(f"Result 2: {jq.get_result(id2)}")
    try:
        print(f"Result 3: {jq.get_result(id3)}")
    except Exception as e:
        print(f"Result 3 failed: {e}")

    jq.shutdown()
