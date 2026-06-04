def process_job(self, job_id, processor):
        with self.lock:
            data = self.jobs.pop(job_id, None)
            if data is None:
                return False

        # Processor runs outside the lock so other jobs can be processed
        result = processor(data)

        with self.lock:
            self.results[job_id] = result
            return True