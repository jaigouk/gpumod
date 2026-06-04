from collections import deque
    from typing import Dict, Optional

    class JobQueue:
        def __init__(self):
            self._queue = deque()
            self._results = {}

        def add_job(self, job_id: str, data: dict) -> str:
            self._queue.append((job_id, data))
            return job_id

        def get_result(self, job_id: str) -> Optional[dict]:
            return self._results.get(job_id)

        def process_jobs(self):
            # Helping the user's example logic
            while self._queue:
                jid, data = self._queue.popleft()
                # Simulate processing
                self._results[jid] = {"status": "completed", "data": data}