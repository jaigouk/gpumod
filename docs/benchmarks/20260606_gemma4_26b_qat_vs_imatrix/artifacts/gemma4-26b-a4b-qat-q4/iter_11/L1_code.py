from collections import deque

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self._queue.append((job_id, data))
        return job_id

    def process_next_job(self):
        """
        Processes the next job in the queue following FIFO order.
        In a real-world scenario, this would involve executing the task logic.
        """
        if not self._queue:
            return

        job_id, data = self._queue.popleft()
        # Simulating task processing by returning the input data in a result envelope
        self._results[job_id] = {"status": "completed", "data": data}

    def get_result(self, job_id: str) -> dict | None:
        return self._results.get(job_id)