import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to synchronize access to shared state
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        # Lock the dictionary update to prevent concurrent modification issues
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # Acquire the lock to perform the check, retrieval, processing, 
        # result storage, and deletion atomically.
        with self.lock:
            if job_id not in self.jobs:
                return False

            data = self.jobs[job_id]

            # Note: The processor runs inside the lock. 
            # This ensures no other thread can pick up this job_id while it is running.
            result = processor(data)

            # Fixed: Race condition resolved. Only one thread can write to results at a time.
            self.results[job_id] = result

            del self.jobs[job_id]
            return True

    def get_result(self, job_id):
        # Lock access to read results safely
        with self.lock:
            return self.results.get(job_id)