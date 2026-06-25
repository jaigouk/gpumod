from collections import deque

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._job_data = {}
        self._results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self._job_data[job_id] = data
        self._queue.append(job_id)
        return job_id

    def process_next(self):
        """
        Processes the next job in the queue following FIFO order.
        In a real-world scenario, this would involve executing a task.
        """
        if not self._queue:
            return None

        job_id = self._queue.popleft()
        data = self._job_data.pop(job_id, None)

        # Simulate processing logic
        result = {"status": "completed", "processed_data": data}
        self._results[job_id] = result
        return result

    def get_result(self, job_id: str) -> dict | None:
        return self._results.get(job_id)