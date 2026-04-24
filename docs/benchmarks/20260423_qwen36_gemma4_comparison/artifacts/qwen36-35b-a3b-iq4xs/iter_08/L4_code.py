import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to manage concurrent access to shared state
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        # Lock to ensure atomicity when adding a job
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # Lock to safely check for job existence and retrieve data.
        # We use pop() to atomically remove the job from the queue.
        # This prevents multiple threads from processing the same job.
        with self.lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id)
        
        # Perform the actual processing outside the lock.
        # This allows other threads to pick up other jobs while this one is running.
        result = processor(data)
        
        # Lock to safely write the result to the results dictionary
        with self.lock:
            self.results[job_id] = result
        
        return True

    def get_result(self, job_id):
        # Lock to safely read from the results dictionary
        with self.lock:
            return self.results.get(job_id)