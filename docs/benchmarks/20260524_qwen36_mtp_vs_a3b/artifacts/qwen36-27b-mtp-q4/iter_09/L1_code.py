from collections import deque
   import threading

   class JobQueue:
       def __init__(self):
           self.queue = deque()
           self.results = {}
           self.lock = threading.Lock()

       def add_job(self, job_id: str, data: dict) -> str:
           with self.lock:
               self.queue.append({"job_id": job_id, "data": data})
           return job_id

       def process_next(self):
           with self.lock:
               if not self.queue:
                   return
               job = self.queue.popleft()
               # Simulate processing
               result = {**job["data"], "status": "completed"}
               self.results[job["job_id"]] = result