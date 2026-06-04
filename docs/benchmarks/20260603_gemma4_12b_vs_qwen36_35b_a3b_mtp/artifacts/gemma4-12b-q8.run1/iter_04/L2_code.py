if job_id not in self.jobs:
            return False
        
        job_data = self.jobs[job_id]
        max_retries = 3
        
        for attempt in range(max_retries + 1):
            try:
                processor(job_data)
                # If it succeeds, we should probably reset retry count or just return True
                return True
            except Exception as e:
                if attempt < max_retries:
                    self.retries[job_id] = attempt + 1
                    backoff = 2 ** attempt # 2^0=1, 2^1=2, 2^2=4
                    # Print/log backoff for simulation
                    pass 
                else:
                    return False