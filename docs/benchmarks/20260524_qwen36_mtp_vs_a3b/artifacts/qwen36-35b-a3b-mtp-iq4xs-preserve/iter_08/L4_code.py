import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        with self.lock:
            if job_id not in self.jobs:
                return False

            data = self.jobs[job_id]
            # Note: processor(data) might take time. If we hold the lock here,
            # it will block other threads from adding/processing jobs.
            # It's usually better to release the lock before calling the processor,
            # but we need to copy the data out first.
            del self.jobs[job_id]
        
        # Process outside the lock to avoid holding it during potentially long operations
        result = processor(data)
        
        with self.lock:
            self.results[job_id] = result
            
        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)