def process_job(self, job_id, processor):
        with self.lock_jobs:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id) # Atomic pop

        # Now the job is ours, no other thread can see it in self.jobs
        result = processor(data)

        with self.lock_results:
            self.results[job_id] = result
        return True