from collections import deque
    from typing import Dict, Any, Optional

    class JobQueue:
        def __init__(self):
            self._queue = deque()
            self._results = {}

        def add_job(self, job_id: str, data: dict) -> str:
            self._queue.append((job_id, data))
            return job_id

        def process_next_job(self, worker_func):
            """
            Internal/Helper method to simulate processing.
            In a real system, a worker would call this.
            """
            if not self._queue:
                return None
            job_id, data = self._queue.popleft()
            result = worker_func(data)
            self._results[job_id] = result
            return result

        def get_result(self, job_id: str) -> Optional[dict]:
            return self._results.get(job_id)