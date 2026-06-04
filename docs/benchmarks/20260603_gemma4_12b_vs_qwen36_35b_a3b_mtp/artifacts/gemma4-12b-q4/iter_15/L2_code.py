def process_job(self, job_id: str, processor: Callable) -> bool:
            if job_id not in self.jobs:
                return False

            max_retries = 3
            for attempt in range(max_retries + 1):
                try:
                    processor(self.jobs[job_id]))
                    self.retry_counts[job_id] = 0 # Reset on success
                    return True
                except Exception as e:
                    if attempt < max_retries:
                        self.retry_counts[job_id] += 1
                        # Logic for backoff (simulated)
                        # delay = 2 ** (self.retry_counts[job_id] - 1)
                        pass
                    else:
                        return False
            return False