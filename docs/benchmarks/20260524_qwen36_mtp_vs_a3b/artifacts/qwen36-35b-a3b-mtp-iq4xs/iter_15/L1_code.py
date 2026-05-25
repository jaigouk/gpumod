from collections import deque
import threading
import time
import uuid

class JobQueue:
    def __init__(self):
        self.queue = deque()
        self.results = {}
        self._lock = threading.Lock()
        self._processor_thread = None
        self._running = False

    def add_job(self, job_id: str, data: dict) -> str:
        with self._lock:
            self.queue.append((job_id, data))
            return job_id

    def get_result(self, job_id: str) -> dict | None:
        with self._lock:
            return self.results.get(job_id)

    def process(self):
        with self._lock:
            if not self.queue:
                return None
            job_id, data = self.queue.popleft()
        # Simulate processing
        time.sleep(0.1)
        result = {"status": "completed", "data": data}
        with self._lock:
            self.results[job_id] = result
        return result

    def process_all(self):
        while self.queue:
            self.process()