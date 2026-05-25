class JobQueue:
         def __init__(self):
             self.jobs = {}
             self.retry_counts = {}
             self.backoff_delays = {}

         def add_job(self, job_id, data):
             self.jobs[job_id] = data
             self.retry_counts[job_id] = 0
             self.backoff_delays[job_id] = []

         def process_job(self, job_id, processor):
             max_retries = 3
             backoff_times = [1, 2, 4]  # seconds
             retries = self.retry_counts.get(job_id, 0)

             for attempt in range(max_retries + 1):  # 0 to 3 -> 4 attempts total? Wait.
                 # Requirement says: "retry up to 3 times". Usually means initial attempt + 3 retries = 4 total, or 3 total attempts?
                 # Let's interpret "retry up to 3 times" as max 3 retries after the first attempt. So attempts = 4.
                 # But often in coding challenges, "retry up to 3 times" means total attempts = 3. I'll stick to 3 retries (4 attempts) or clarify by code structure. Actually, I'll do 3 retries total (attempts 0,1,2,3) or just 3 total attempts? Let's assume max_retries = 3 means up to 3 retries. I'll use a loop for retries in range(3).
                 # Let's re-read: "retry up to 3 times". I'll implement 3 retries.
                 pass