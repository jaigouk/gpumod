if job_id not in self.jobs: # Check
        return False
    data = self.jobs[job_id]   # Read
    result = processor(data)   # Work (Don't hold lock here ideally, but logic dictates we own the job now)
    self.results[job_id] = result # Write
    del self.jobs[job_id]       # Delete