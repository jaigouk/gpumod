import collections

   class JobQueue:
       def __init__(self):
           self._queue = collections.deque()
           self._results = {}
           self._jobs = {}

       def add_job(self, job_id: str, data: dict) -> str:
           if job_id in self._results or job_id in self._jobs:
               raise ValueError("Duplicate job_id")
           self._jobs[job_id] = data
           self._queue.append(job_id)
           return job_id

       def get_result(self, job_id: str) -> dict | None:
           return self._results.get(job_id, None)

       def process_next(self) -> dict | None:
           if not self._queue:
               return None
           job_id = self._queue.popleft()
           data = self._jobs.pop(job_id)
           # Simulate processing
           result = {"status": "completed", "processed_data": data}
           self._results[job_id] = result
           return result