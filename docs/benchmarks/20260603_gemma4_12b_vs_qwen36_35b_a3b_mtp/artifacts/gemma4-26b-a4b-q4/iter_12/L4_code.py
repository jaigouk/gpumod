with self._lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id) # Atomically take ownership

        # Process outside the lock to allow concurrency!
        result = processor(data)

        with self._lock:
            self.results[job_id] = result
        return True