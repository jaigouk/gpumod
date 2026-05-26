def process_job(self, job_id, processor):
    if job_id not in self.jobs:
        return False

    data = self.jobs[job_id]
    result = processor(data)

    # BUG: Race condition here - multiple threads can write simultaneously
    self.results[job_id] = result

    del self.jobs[job_id]
    return True