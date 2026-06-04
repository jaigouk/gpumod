max_retries = 3
        try:
            while True:
                try:
                    processor(self.jobs[job_id])
                    return True
                except Exception:
                    retries = self.retry_counts.get(job_id, 0)
                    if retries >= max_retries:
                        return False
                    self.retry_counts[job_id] = retries + 1
                    # delay = 2 ** retries (1, 2, 4)
                    # (No actual sleep needed per requirements)