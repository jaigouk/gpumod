import threading
from collections import deque

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._results = {}
        self._condition = threading.Condition()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def add_job(self, job_id: str, data: dict) -> str:
        with self._condition:
            self._queue.append((job_id, data))
            self._condition.notify()
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        with self._condition:
            return self._results.get(job_id)

    def _worker_loop(self):
        while True:
            with self._condition:
                while not self._queue:
                    self._condition.wait()
                job_id, data = self._queue.popleft()

            # Simulate job processing
            result = {"status": "completed", "data": data}

            with self._condition:
                self._results[job_id] = result