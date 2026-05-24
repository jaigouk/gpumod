import collections
       from typing import Optional

       class JobQueue:
           def __init__(self):
               self._queue = collections.deque()
               self._results = {}

           def add_job(self, job_id: str, data: dict) -> str:
               self._queue.append({"job_id": job_id, "data": data})
               return job_id

           def process_pending(self):
               while self._queue:
                   job = self._queue.popleft()
                   # Simulate processing
                   result = {"job_id": job["job_id"], "status": "completed", "result": job["data"]}
                   self._results[job["job_id"]] = result

           def get_result(self, job_id: str) -> Optional[dict]:
               return self._results.get(job_id)