if job_id not in self.jobs:
                return False
            data = self.jobs[job_id]
            # ... processing ...
            self.results[job_id] = result
            del self.jobs[job_id]