def process_job(self, job_id, processor):
        if job_id not in self.jobs:
            return False
        
        job_data = self.jobs[job_id]
        max_retries = 3
        
        for attempt in range(max_retries + 1):
            try:
                processor(job_data)
                return True
            except Exception as e:
                if attempt < max_retries:
                    # Backoff logic
                    delay = 2 ** attempt # 2^0=1, 2^1=2, 2^2=4
                    self.retries[job_id] = attempt + 1
                    # Simulation of sleep: print(f"Retrying {job_id} in {delay}s")
                else:
                    return False