import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Lock to synchronize access to shared state
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # 1. Safely check existence and remove job from queue
        with self.lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id)
            
        # 2. Process outside the lock to allow concurrency
        result = processor(data)
        
        # 3. Safely store the result
        with self.lock:
            self.results[job_id] = result
            
        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)