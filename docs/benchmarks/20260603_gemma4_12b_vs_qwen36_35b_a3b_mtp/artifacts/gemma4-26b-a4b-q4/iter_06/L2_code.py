retries = 0
        max_retries = 3
        while retries <= max_retries:
            try:
                processor(self.jobs[job_id])
                return True
            except Exception:
                if retries == max_retries:
                    return False
                delay = 2 ** retries # 1, 2, 4...
                # simulation: pass or log delay
                retries += 1