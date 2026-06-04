from collections import deque
    from typing import Dict, Any, Optional

    class JobQueue:
        def __init__(self):
            self._queue = deque()
            self._results = {}

        def add_job(self, job_id: str, data: dict) -> str:
            self._queue.append((job_id, data))
            return job_id

        def get_result(self, job_id: str) -> Optional[dict]:
            return self._results.get(job_id)

        def complete_job(self, job_id: str, result: dict):
            """Helper to simulate job completion """
            self._results[job_id] = result