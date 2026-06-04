def process_job(self, job_id, Processor):
        if job_id not in self.jobs: # Check 1
            return False

        data = self.jobs[job_id] # Access 1
        result = processor(data)

        # BUG: Race condition here
        self.results[job_id] = result # Write 1
        del self.jobs[jobie] # Write 2
        return True