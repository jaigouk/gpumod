with self.lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id) # pop is atomic relative to the lock

        # Execute outside the lock to allow parallelism
        result = processor(data)

        with self.lock:
            self.results[job_id] = result
        return True