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
                return None
            job_id, data = self._queue.popleft()
            # Mock processing: just return the data back or something simple
            result = {"status": "completed", "original_data": data}
            self._results[job_id] = result
            return result

        def get_result(self, job_id: str) -> dict | None:
            return self._results.get(job_id)