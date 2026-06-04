data = self.jobs[job_id]
    result = processor(data) # Expensive operation
    self.results[job_id] = result
    del self.jobs[job_id]