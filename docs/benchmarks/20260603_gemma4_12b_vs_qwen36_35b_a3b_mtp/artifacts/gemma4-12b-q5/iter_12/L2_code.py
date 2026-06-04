max_retries = 3
        job = self.jobs.get(job_id)
        for attempt in range(max_retries + 1):
            try:
                processor(job["data"])
                return True
            except Exception:
                if attempt < max_retries:
                    self.jobs[job_id]["retries"] += 1
                    # simulate backoff
                    delay = 2 ** (self.jobs[job_id]["retries"] - 1)
                    # print/log delay? No, requirement says "can be simulated".
                else:
                    return False