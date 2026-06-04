def process_job(self, job_id, processor):
        with self.lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id) # Pop is atomic and handles removal

        result = processor(data) # Run outside lock!

        with self.lock:
            self.results[job_id] = result
        return True