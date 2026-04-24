import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # 1. Add a lock to protect shared state
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        # 2. Protect dictionary modification with the lock
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # Extract data and remove the job under lock. 
        # This prevents another thread from picking up the same job concurrently.
        with self.lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs[job_id]
            del self.jobs[job_id]
            
        # 3. Process outside the lock to avoid blocking other threads during 
        # potentially long-running tasks.
        result = processor(data)
        
        # 4. Store the result under lock to fix the race condition on writes
        with self.lock:
            self.results[job_id] = result
            
        return True

    def get_result(self, job_id):
        # 5. Protect dictionary reads with the lock
        with self.lock:
            return self.results.get(job_id)