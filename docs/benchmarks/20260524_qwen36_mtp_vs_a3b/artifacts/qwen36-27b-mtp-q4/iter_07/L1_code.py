from collections import deque
import threading

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._results = {}
        self._lock = threading.Lock()
        self._running = False
        self._worker = None

    def add_job(self, job_id: str, data: dict) -> str:
        with self._lock:
            self._queue.append((job_id, data))
            if not self._running:
                self._running = True
                self._worker = threading.Thread(target=self._process_loop, daemon=True)
                self._worker.start()
        return job_id

    def _process_loop(self):
        while self._running or self._queue:
            with self._lock:
                if self._queue:
                    job_id, data = self._queue.popleft()
                else:
                    break
            # Simulate processing
            result = {"status": "completed", "data": data}
            with self._lock:
                self._results[job_id] = result
        self._running = False

    def get_result(self, job_id: str) -> dict | None:
        with self._lock:
            return self._results.get(job_id)