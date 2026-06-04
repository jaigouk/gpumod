def process_job(self, job_id, processor):
            with self.lock:
                if job_id not in self.jobs:
                    return False
                data = self.jobs.pop(job_id)

            # Run processor outside the lock to allow concurrency
            result = processor(data)

            with self.lock:
                self.results[job_id] = result
            return True