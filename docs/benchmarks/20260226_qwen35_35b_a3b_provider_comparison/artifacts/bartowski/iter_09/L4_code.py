import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        """Add a job to the queue in a thread-safe manner."""
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        """
        Process a job in a thread-safe manner.
        The actual processing occurs outside the lock to avoid blocking other threads.
        """
        with self.lock:
            # Atomically check and retrieve the job
            if job_id not in self.jobs:
                return False
            data = self.jobs[job_id]
            # Remove the job from the queue to prevent reprocessing
            del self.jobs[job_id]

        # Perform the actual processing outside the lock
        result = processor(data)

        # Store the result in a thread-safe manner
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        """Retrieve the result of a job in a thread-safe manner."""
        with self.lock:
            return self.results.get(job_id)