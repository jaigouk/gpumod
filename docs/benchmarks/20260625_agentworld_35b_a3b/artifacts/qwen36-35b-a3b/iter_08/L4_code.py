import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Create a lock to synchronize access to shared state
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        # Lock ensures that adding a job is thread-safe
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # We need to atomically check for the job, retrieve it, and remove it.
        # If we don't remove it here, another thread might pick it up while we are processing.
        with self.lock:
            if job_id not in self.jobs:
                return False

            data = self.jobs[job_id]
            # Remove the job from the queue immediately to prevent double processing
            del self.jobs[job_id]

        # Process the job outside the lock. 
        # This allows other threads to add jobs or check results while this one runs.
        # Note: If processor(data) raises an exception, the job is lost (deleted from queue)
        # but no result is stored. This is standard behavior for a fire-and-forget queue,
        # though production systems might wrap this in a try/finally block.
        result = processor(data)

        # Lock ensures that writing the result is thread-safe
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        # Lock ensures reading the result is safe
        with self.lock:
            return self.results.get(job_id)