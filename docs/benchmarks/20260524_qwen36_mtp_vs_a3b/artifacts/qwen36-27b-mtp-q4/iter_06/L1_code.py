from collections import deque
   from typing import Dict, Optional

   class JobQueue:
       def __init__(self):
           self._queue: deque = deque()
           self._results: Dict[str, dict] = {}

       def add_job(self, job_id: str, data: dict) -> str:
           self._queue.append({"job_id": job_id, "data": data})
           return job_id

       def process_jobs(self):
           while self._queue:
               job = self._queue.popleft()
               job_id = job["job_id"]
               # Simulate processing
               self._results[job_id] = {"status": "completed", "input": job["data"]}

       def get_result(self, job_id: str) -> dict | None:
           return self._results.get(job_id)