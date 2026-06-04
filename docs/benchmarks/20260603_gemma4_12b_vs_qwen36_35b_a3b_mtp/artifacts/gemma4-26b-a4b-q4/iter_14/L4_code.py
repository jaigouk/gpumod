with self.lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id) # Atomic removal

        result = processor(data) # Slow part, outside lock

        with self.lock:
            self.results[job_id] = result
        return True