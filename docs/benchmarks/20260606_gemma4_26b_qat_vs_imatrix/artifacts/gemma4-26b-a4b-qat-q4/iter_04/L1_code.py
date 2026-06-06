import collections
import threading

class JobQueue:
    def __init__(self):
        self._queue = collections.deque()
        self._results = {}
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

    def add_job(self, job_id: str, data: dict) -> str:
        with self._condition:
            self._queue.append((job_id, data))
            self._condition.notify()
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        with self._lock:
            return self._results.get(job_id)

    def _worker(self) -> None:
        while True:
            with self._condition:
                while not self._queue:
                    self._condition.wait()
                job_id, data = self._queue.popleft()

            # Simulate processing the job
            result = {"status": "completed", "input": data}

            with self._lock:
                self._results[job_id] = result