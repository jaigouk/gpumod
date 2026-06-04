with self.lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id) # Pop is atomic for the dict, but we need the lock for the check+pop sequence
        
        result = processor(data)
        
        with self.lock:
            self.results[job_id] = result