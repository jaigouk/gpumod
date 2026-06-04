with self.lock:
        if job_id not in self.jobs:
            return False
        data = self.jobs.pop(job_id) # Remove it immediately so no one else can grab it
    
    result = processor(data)

    with self.lock:
        self.results[job_id] = result