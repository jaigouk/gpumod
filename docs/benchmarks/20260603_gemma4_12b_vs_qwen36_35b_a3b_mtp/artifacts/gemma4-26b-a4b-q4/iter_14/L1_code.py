from collections import deque

    class JobQueue:
        def __init__(self):
            self._queue = deque()
            self._results = {}

        def add_job(self, job_id: str, data: dict) -> str:
            self._queue.append((job_id, data))
            return job_id

        def process_next_job(self):
            if not self._queue:
                return
            job_id, data = self._queue.popleft()
            # Simulating processing: just returning the data as the result
            self._results[job_id] = {"status": "completed", "output": data}

        def get_result(self, job_id: str) -> dict | None:
            return self._results.get(job_id)