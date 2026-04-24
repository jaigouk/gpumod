import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to synchronize access to shared state
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        # Protect the write to the jobs dictionary
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # Acquire lock to safely inspect and modify the jobs dictionary
        with self.lock:
            if job_id not in self.jobs:
                return False
            
            # Atomically retrieve data and remove the job from the queue.
            # Removing it here (claiming it) prevents multiple threads from 
            # processing the same job_id simultaneously.
            data = self.jobs.pop(job_id)
        
        # Perform the actual processing outside the lock.
        # This allows other threads to add or process other jobs concurrently.
        result = processor(data)

        # Acquire lock again to safely store the result
        with self.lock:
            self.results[job_id] = result
        
        return True

    def get_result(self, job_id):
        # Protect the read from the results dictionary
        with self.lock:
            return self.results.get(job_id)