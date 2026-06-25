import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to protect shared state
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        # Lock is required to ensure atomic addition to the dictionary
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # Critical Section 1: Check and Retrieve Job
        with self.lock:
            if job_id not in self.jobs:
                return False

            # Retrieve and remove the job. 
            # Removing it here prevents other threads from picking up the same job.
            # This effectively "pops" the job from the queue.
            data = self.jobs.pop(job_id)

        # Process the job outside the lock.
        # This allows other threads to add jobs or check results while this job is processing.
        # Note: If the processor raises an exception, the job is considered lost (removed from queue).
        result = processor(data)

        # Critical Section 2: Store Result
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        # Lock is required to ensure consistent read of results
        with self.lock:
            return self.results.get(job_id)