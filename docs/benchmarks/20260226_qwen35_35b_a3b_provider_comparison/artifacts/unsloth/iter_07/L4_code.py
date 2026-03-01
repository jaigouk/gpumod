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
            # Process the job
            result = processor(data)
            # Store result
            self.results[job_id] = result
            # Remove job from queue
            del self.jobs[job_id]
        
        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)