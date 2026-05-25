with self.lock:
            self.jobs[job_id] = data
        return job_id