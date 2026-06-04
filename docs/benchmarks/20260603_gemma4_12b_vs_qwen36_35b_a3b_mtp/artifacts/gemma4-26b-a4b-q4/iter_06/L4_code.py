def process_job(self, job_id, processor):
        if job_id not in self.jobs: # Check
            return False

        data = self.jobs[job_id] # Access
        result = processor(data) # Long-running operation (unlocked)

        # BUG: Race condition here - multiple threads can write simultaneously
        self.results[job_id] = result # Write

        del self.jobs[job_id] # Delete
        return True