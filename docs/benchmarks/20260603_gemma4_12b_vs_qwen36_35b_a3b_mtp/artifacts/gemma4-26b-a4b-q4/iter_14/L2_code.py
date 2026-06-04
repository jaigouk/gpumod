attempts = 0
            max_retries = 3
            while attempts <= max_retries:
                try:
                    processor(self.jobs[job_id])
                    return True
                except Exception:
                    if attempts == max_retries:
                        return False
                    attempts += 1
                    delay = 2**(attempts - 1) # 1, 2, 4
                    # track delay somehow? Requirement says "can be simulated"