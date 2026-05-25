import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        self.lock = threading.Lock() # Create the lock

    def add_job(self, job_id, data):
        with self.lock: # Protect adding
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # We need to check existence and remove atomically to prevent double processing
        with self.lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id) # pop is atomic and safe here under lock

        # Process outside the lock to allow concurrency
        # Note: If processor modifies shared state, it needs its own locking, 
        # but that's outside the scope of fixing the Queue's race condition.
        result = processor(data)

        # Store result safely
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self.lock: # Protect reading results
            return self.results.get(job_id)