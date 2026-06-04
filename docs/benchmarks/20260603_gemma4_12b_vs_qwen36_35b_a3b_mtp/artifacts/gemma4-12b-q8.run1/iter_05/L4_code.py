def process_job(self, job_id, processor):
        with self.lock:
            if job_id not in self.jobs:
                return False
            # Atomically remove the job so no other thread can pick it up
            data = self.jobs.pop(job_id)

        # Now we can process outside the lock to allow concurrency
        result = processor(data)

        with self.lock:
            self.results[job_id] = result
            return True