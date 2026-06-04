def process_job(self, job_id, Processor):
        if job_id not in self.jobs: # Check
            return False

        data = self.jobs[job_id]     # Access
        result = processor(data)      # Execution (long running)

        # BUG: Race condition here
        self.results[job_id] = result # Update 1
        del self.jobs[job_id]          # Update 2
        return True