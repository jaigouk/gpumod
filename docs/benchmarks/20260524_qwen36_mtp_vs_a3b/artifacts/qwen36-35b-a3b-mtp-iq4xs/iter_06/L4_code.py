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
            # Note: running the processor outside the lock is usually better to avoid blocking other jobs,
            # but we need to remove the job from self.jobs before processing to prevent duplicate processing.
            # However, if we remove it inside the lock, we hold the lock while processing, which might be slow.
            # Let's think about standard queue patterns. Usually, you pop the job inside the lock, then process outside.
            del self.jobs[job_id]
            
        # Process outside the lock to avoid holding it during potentially long operations
        result = processor(data)
        
        with self.lock:
            self.results[job_id] = result
            
        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)