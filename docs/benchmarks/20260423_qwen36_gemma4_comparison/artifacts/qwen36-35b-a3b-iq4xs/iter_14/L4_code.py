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
            # pop() atomically retrieves and removes the job, preventing TOCTOU races
            data = self.jobs.pop(job_id)
            
        # Release lock before processing to allow true concurrent execution
        result = processor(data)
        
        with self.lock:
            self.results[job_id] = result
        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)