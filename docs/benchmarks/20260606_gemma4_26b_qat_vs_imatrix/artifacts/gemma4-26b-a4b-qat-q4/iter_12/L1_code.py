from collections import deque

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self._queue.append((job_id, data))
        return job_id

    def process_next(self) -> str | None:
        """
        Processes the next job in the queue in FIFO order.
        """
        if not self._queue:
            return None

        job_id, data = self._queue.popleft()
        # Simulate processing logic
        self._results[job_id] = {"status": "completed", "result": data}
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        return self._results.get(job_id)