attempts = 0
        max_retries = 3
        while attempts <= max_retries:
            try:
                processor(self.jobs[job_id])
                return True
            except Exception:
                attempts += 1
                if attempts > max_retries:
                    return False
                delay = 2**(attempts - 1) # 1, 2, 4
                # Simulation: log delay
                # time.sleep(delay)