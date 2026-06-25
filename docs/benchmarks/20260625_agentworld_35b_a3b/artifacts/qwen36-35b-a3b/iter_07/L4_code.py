import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to protect shared state
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        # Lock while adding a job to ensure dictionary consistency
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # Lock to atomically check for the job and remove it from the queue.
        # Using pop() ensures that if a thread retrieves the job, it is 
        # immediately removed, preventing other threads from processing it.
        with self.lock:
            if job_id not in self.jobs:
                return False
            # pop() removes the item and returns it safely
            data = self.jobs.pop(job_id)

        # Process the data outside the lock to allow concurrency.
        # We have already secured the data and removed the job from the queue.
        result = processor(data)

        # Lock to safely write the result to the results dictionary
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        # Lock while reading results to ensure thread-safe access
        with self.lock:
            return self.results.get(job_id)