class JobQueue:
       def __init__(self):
           self.jobs = {}  # job_id -> data
           self.retry_counts = {}  # job_id -> current retry count
           self.backoff_delays = {}  # job_id -> list of delays used

       def add_job(self, job_id: str, data):
           self.jobs[job_id] = data
           self.retry_counts[job_id] = 0
           self.backoff_delays[job_id] = []

       def process_job(self, job_id: str, processor: Callable) -> bool:
           if job_id not in self.jobs:
               return False

           max_retries = 3
           # Try initial attempt + up to 3 retries = 4 total attempts?
           # Wait, requirement says "retry up to 3 times". Usually means 1 initial + 3 retries = 4 attempts total.
           # Let's implement exactly 3 retries after the first failure.
           # So total attempts = 4. Or maybe just 3 total attempts? I'll stick to 3 retries (4 attempts total) or exactly 3 attempts total.
           # The wording "retry up to 3 times" typically means 3 retries after the first failure.
           # I'll implement a loop that attempts up to 4 times (1 initial + 3 retries), or I can just track retries and stop after 3.
           # Let's use a for loop with range(4) for attempts, or explicitly handle retries.
           # Actually, simpler: attempt count from 0 to 3 (4 attempts total), or retry count from 0 to 2 (3 retries).
           # I'll go with: max_attempts = 4 (1 initial + 3 retries)
           # Or I'll just do: for attempt in range(4): ... if attempt > 0: apply backoff
           # Wait, the requirement says "retry up to 3 times". I'll implement exactly that.
           # Let's track retry count separately.
           # I'll reset retry count for this job before processing? Or keep it persistent? The requirement says "Track retry count for each job". I'll reset it on each process_job call or keep it cumulative? Usually, it's per execution. I'll reset it or just track it during this call. I'll store it in self.retry_counts[job_id] and update it.
           # Actually, I'll just track it internally during the call and update the stored count.
           # Let's keep it simple: self.retry_counts[job_id] = 0 at start of process_job, increment on failure, store delays.