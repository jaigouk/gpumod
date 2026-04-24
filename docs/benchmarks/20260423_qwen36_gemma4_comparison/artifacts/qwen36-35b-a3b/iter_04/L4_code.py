import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to protect shared state
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        # Lock the dictionary while adding a job
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        with self.lock:
            if job_id not in self.jobs:
                return False
            
            # Retrieve the data and remove it from the queue immediately.
            # We must do this while holding the lock to prevent another thread
            # from picking up the same job (race condition).
            data = self.jobs.pop(job_id)
        
        # Perform the processing outside the lock.
        # This allows other threads to pick up different jobs while this one runs.
        result = processor(data)
        
        with self.lock:
            # Lock again before writing the result to the shared dictionary
            self.results[job_id] = result
            
        return True

    def get_result(self, job_id):
        # Lock the dictionary while reading the result
        with self.lock:
            return self.results.get(job_id)