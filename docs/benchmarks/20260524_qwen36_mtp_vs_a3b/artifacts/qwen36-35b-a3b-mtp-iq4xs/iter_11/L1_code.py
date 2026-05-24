import collections

  class JobQueue:
      def __init__(self):
          self._queue = collections.deque()
          self._jobs = {}

      def add_job(self, job_id: str, data: dict) -> str:
          self._queue.append(job_id)
          self._jobs[job_id] = {"data": data, "status": "pending", "result": None}
          return job_id

      def get_result(self, job_id: str) -> dict | None:
          if job_id in self._jobs:
              return self._jobs[job_id].get("result")
          return None

      def process_jobs(self):
          while self._queue:
              job_id = self._queue.popleft()
              # Simulate processing
              self._jobs[job_id]["status"] = "completed"
              self._jobs[job_id]["result"] = {"output": f"processed_{self._jobs[job_id]['data']}"}