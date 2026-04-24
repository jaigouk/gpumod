import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to manage concurrent access to shared state
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        # Lock to safely add jobs to the shared dictionary
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # Lock to ensure that checking, processing, and updating state 
        # are atomic. This prevents multiple threads from processing 
        # the same job or overwriting results.
        with self.lock:
            if job_id not in self.jobs:
                return False

            data = self.jobs[job_id]
            
            # Note: The processor is executed while holding the lock.
            # This ensures data consistency. In high-performance scenarios,
            # one might pop the job from the queue before processing to 
            # allow other jobs to run, but for this specific fix, 
            # locking the whole method is the most robust solution 
            # to prevent race conditions on the job state.
            result = processor(data)

            self.results[job_id] = result
            del self.jobs[job_id]
            return True

    def get_result(self, job_id):
        # Lock to safely read from the shared results dictionary
        with self.lock:
            return self.results.get(job_id)