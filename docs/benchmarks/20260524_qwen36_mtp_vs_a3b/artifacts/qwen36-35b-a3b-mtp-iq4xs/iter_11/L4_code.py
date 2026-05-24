def add_job(self, job_id, data):
        with self.lock:
            self.jobs[job_id] = data
            return job_id