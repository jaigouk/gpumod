from collections import deque

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self._queue.append((job_id, data))
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        while self._queue:
            current_id, current_data = self._queue.popleft()

            # Simulate the processing of the job
            self._results[current_id] = {
                "status": "completed",
                "input_data": current_data
            }

            # If we reached the requested job, stop processing the queue
            if current_id == job_id:
                break

        return self._results.get(job_id)