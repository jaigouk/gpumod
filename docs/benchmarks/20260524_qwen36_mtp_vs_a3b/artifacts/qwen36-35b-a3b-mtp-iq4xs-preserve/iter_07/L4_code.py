def process_job(self, job_id, processor):
        with self.lock:
            if job_id not in self.jobs: return False
            data = self.jobs.pop(job_id) # pop is atomic
            self.results[job_id] = processor(data)
        return True