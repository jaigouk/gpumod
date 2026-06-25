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
            # Check if job exists
            if job_id not in self.jobs:
                return False

            # Atomically retrieve and remove the job from the queue.
            # Using pop() ensures that no other thread can process this job_id
            # while we are holding the lock.
            data = self.jobs.pop(job_id)

        # Perform the processing outside the lock.
        # This allows other threads to add or process other jobs concurrently.
        result = processor(data)

        with self.lock:
            # Lock again to safely store the result
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self.lock:
            # Lock to safely read the result
            return self.results.get(job_id)