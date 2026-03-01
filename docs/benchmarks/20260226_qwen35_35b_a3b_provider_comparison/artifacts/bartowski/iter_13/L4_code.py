import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        with self._lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        with self._lock:
            if job_id not in self.jobs:
                return False
            
            # Copy data before releasing lock to avoid blocking during processing
            data = self.jobs[job_id]
            del self.jobs[job_id]
        
        # Process outside the lock to allow concurrent job processing
        result = processor(data)
        
        # Store result under lock for thread safety
        with self._lock:
            self.results[job_id] = result
        
        return True

    def get_result(self, job_id):
        with self._lock:
            return self.results.get(job_id)