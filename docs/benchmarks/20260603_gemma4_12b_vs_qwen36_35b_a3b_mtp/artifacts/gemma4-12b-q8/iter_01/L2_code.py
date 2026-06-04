class JobQueue:
            def __init__(self):
                self.jobs = {}
                self.retry_counts = {}

            def add_job(self, job_id, data):
                self.jobs[job_id] = data
                self.retry_counts[job_id] = 0

            def process_job(self, job_id, processor):
                data = self.jobs.get(job_id)
                if not data:
                    return False
                
                max_retries = 3
                for attempt in range(max_retries + 1): # Attempt 0 is first try, then retries
                    # Wait, "retry up to 3 times" means 4 total attempts?
                    # Usually: Attempt 1 (fail) -> Retry 1 (fail) -> Retry 2 (fail) -> Retry 3 (fail) -> Fail.
                    # Total attempts = 1 + max_retries = 4.
                    try:
                        processor(data)
                        return True
                    except Exception:
                        if attempt < max_retries:
                            self.retry_counts[job_id] += 1
                            delay = 2 ** (self.retry_counts[job_id] - 1)
                            print(f"Retry {self.retry_counts[job_id]} for {job_id} after {delay}s")
                        else:
                             return False