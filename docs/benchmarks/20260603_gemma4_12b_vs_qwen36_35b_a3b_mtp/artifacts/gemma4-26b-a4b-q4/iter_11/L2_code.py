data = self.jobs[job_id]
        retries = 0
        max_retries = 3
        while retries <= max_retries:
            try:
                processor(data)
                return True
            except Exception:
                if retries == max_retries:
                    return False
                # Calculate delay: 1, 2, 4
                delay = 2**retries # 2^0=1, 2^1=2, 2^2=4
                # track delay/retry (as per requirement)
                retries += 1
                # simulating delay