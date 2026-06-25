import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to protect shared state
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        # Lock access to the jobs dictionary to prevent race conditions
        # when adding jobs while others are being processed.
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # Lock to safely check existence and retrieve the job.
        # We also remove the job from the queue here to prevent
        # multiple threads from processing the same job ID simultaneously.
        with self.lock:
            if job_id not in self.jobs:
                return False
            # pop() retrieves the value and removes the key atomically
            data = self.jobs.pop(job_id)

        # Execute the processor OUTSIDE the lock.
        # Holding a lock while running external code (the processor) 
        # blocks other threads from using the queue unnecessarily.
        result = processor(data)

        # Lock to safely write the result to the results dictionary.
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        # Lock to safely read from the results dictionary
        with self.lock:
            return self.results.get(job_id)