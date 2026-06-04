def process_job(self, job_id, processor):
        with self.lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id) # Pop is atomic-ish, but we have the lock anyway

        result = processor(data)

        with self.lock:
            self.results[job_id] = result
        return True