class JobQueue:
       def __init__(self):
           self.jobs = {}
           self.retry_counts = {}

       def add_job(self, job_id: str, data: dict):
           self.jobs[job_id] = data
           self.retry_counts[job_id] = 0