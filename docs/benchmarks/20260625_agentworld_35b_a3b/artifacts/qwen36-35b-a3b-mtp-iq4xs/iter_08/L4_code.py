import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to manage concurrent access to shared state
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        # Lock ensures that adding a job is thread-safe
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # Lock to atomically check for the job and remove it from the queue.
        # This prevents multiple threads from picking up the same job ID.
        with self.lock:
            if job_id not in self.jobs:
                return False
            # Using pop to retrieve and remove the job in one atomic step
            data = self.jobs.pop(job_id)

        # Execute the processor outside the lock.
        # This is crucial for performance: it allows other threads to add 
        # or process different jobs while this one is running.
        result = processor(data)

        # Lock to safely write the result to the shared results dictionary
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        # Lock ensures we read a consistent state of the results dictionary
        with self.lock:
            return self.results.get(job_id)