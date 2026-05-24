if job_id not in self.jobs:
            return False
        data = self.jobs[job_id]
        # ... processing ...
        del self.jobs[job_id]