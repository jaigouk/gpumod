from collections import deque

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._pending_data = {}
        self._results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self._queue.append(job_id)
        self._pending_data[job_id] = data
        return job_id

    def process_next_job(self):
        """
        Processes the next job in the queue in FIFO order.
        """
        if not self._queue:
            return

        job_id = self._queue.popleft()
        data = self._pending_data.pop(job_id, None)

        if data is not None:
            # Simulate processing by returning the data back in a result dictionary
            self._results[job_id] = {"status": "completed", "result": data}

    def get_result(self, job_id: str) -> dict | None:
        return self._results.get(job_id)