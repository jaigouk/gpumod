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

            # Copy the data while holding the lock
            data = self.jobs[job_id]
            
            # Process the job (outside lock to avoid blocking other threads)
            result = processor(data)
            
            # Store result and remove job while still holding the lock
            self.results[job_id] = result
            del self.jobs[job_id]
        
        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)