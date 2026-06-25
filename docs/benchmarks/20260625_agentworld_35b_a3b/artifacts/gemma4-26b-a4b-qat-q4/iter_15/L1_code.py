from collections import deque

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._results = {}
        self._pending_jobs = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self._pending_jobs[job_id] = data
        self._queue.append(job_id)
        return job_id

    def process_next_job(self):
        """
        Processes the next job in the queue in FIFO order.
        """
        if not self._queue:
            return None

        job_id = self._queue.popleft()
        data = self._pending_jobs.pop(job_id)

        # Simulated processing: wrapping the data in a result dictionary
        result = {"status": "completed", "output": data}
        self._results[job_id] = result
        return result

    def get_result(self, job_id: str) -> dict | None:
        return self._results.get(job_id)