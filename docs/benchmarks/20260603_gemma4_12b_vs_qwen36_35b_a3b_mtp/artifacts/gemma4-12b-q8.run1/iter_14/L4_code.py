if job_id not in self.jobs: # Check
        return False
    data = self.jobs[job_id]    # Act
    ...
    del self.jobs[job_id]        # Act