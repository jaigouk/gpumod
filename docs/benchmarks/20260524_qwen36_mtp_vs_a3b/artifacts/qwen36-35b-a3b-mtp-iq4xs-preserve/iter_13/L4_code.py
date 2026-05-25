def process_job(self, job_id, processor):
            if job_id not in self.jobs: # Check 1
                return False

            data = self.jobs[job_id]   # Read 1
            result = processor(data)   # Work
            self.results[job_id] = result # Write 1 (The specific comment points here)
            del self.jobs[job_id]      # Delete 1
            return True