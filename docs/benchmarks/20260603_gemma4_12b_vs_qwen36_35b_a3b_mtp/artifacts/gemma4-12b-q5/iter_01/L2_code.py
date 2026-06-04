def process_job(self, job_id: str, processor: Callable) -> bool:
            if job_id not in self.jobs:
                return False
            
            job_data = self.jobs[job_id]
            max_retries = 3
            backoffs = [1, 2, 4]
            
            for attempt in range(max_retries + 1):
                try:
                    processor(job_data["data"])
                    return True
                except Exception:
                    if attempt < max_retries:
                        # Simulation: we could track the "next allowed time"
                        # but the prompt says "can be simulated". 
                        # I'll just track the count.
                        job_data["retries"] += 1
                    else:
                        return False