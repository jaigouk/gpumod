import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        # Protect the dictionary modification
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # 1. Acquire lock to safely check, retrieve, and remove the job
        with self.lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs[job_id]
            del self.jobs[job_id]
        
        # 2. Process the job OUTSIDE the lock to allow concurrency
        # The lock is released here, allowing other threads to access the queue
        result = processor(data)

        # 3. Acquire lock to safely write the result
        with self.lock:
            self.results[job_id] = result
            
        return True

    def get_result(self, job_id):
        # Protect the read operation for consistency
        with self.lock:
            return self.results.get(job_id)