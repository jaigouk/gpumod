with self.lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id) # pop is atomic with respect to the dict, but we need the lock for the check+pop logic

        result = processor(data)

        with self.lock:
            self.results[job_id] = result