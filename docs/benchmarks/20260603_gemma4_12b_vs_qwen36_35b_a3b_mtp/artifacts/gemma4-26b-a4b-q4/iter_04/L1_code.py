from collections import deque
    from typing import Dict, Optional, Any

    class JobQueue:
        def __init__(self):
            self._queue = deque()
            self._jobs = {}

        def add_job(self, job_id: str, data: dict) -> str:
            self._queue.append(job_id)
            self._jobs[job_id] = {
                "data": data,
                "result": None,
                "completed": False
            }
            return job_id

        def process_next(self):
            if not self._queue:
                return

            job_id = self._queue.popleft()
            job = self._jobs[job_id]

            # Simulating processing: just return the data back or modify it
            # In a real scenario, this would be a task runner.
            # For this implementation, we'll just "complete" it.
            job["result"] = {"status": "completed", "processed_data": job["data"]}
            job["completed"] = True

        def get_result(self, job_id: str) -> Optional[dict]:
            job = self._jobs.get(job_id)
            if job and job["completed"]:
                return job["result"]
            return None