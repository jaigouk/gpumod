if job_id not in self.jobs:
        return False
    data = self.jobs[job_id]
    result = processor(data)
    self.results[job_id] = result # BUG
    del self.jobs[job_id]