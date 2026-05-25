class JobQueue:
       def __init__(self):
           self.queue = []  # Stores job_ids in FIFO order
           self.data = {}   # Stores job data
           self.results = {} # Stores results
       
       def add_job(self, job_id: str, data: dict) -> str:
           if job_id in self.data:
               raise ValueError(f"Job {job_id} already exists")
           self.queue.append(job_id)
           self.data[job_id] = data
           return job_id
       
       def get_result(self, job_id: str) -> dict | None:
           return self.results.get(job_id, None)
       
       def process(self):
           if not self.queue:
               return None
           job_id = self.queue.pop(0)
           # Simulate processing
           # In a real queue, this would execute the task
           self.results[job_id] = {"status": "completed", "data": self.data[job_id]}
           del self.data[job_id]
           return self.results[job_id]