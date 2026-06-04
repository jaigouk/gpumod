import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to synchronize access to shared dictionaries
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, Processor):
        # Step 1: Atomically "claim" the job.
        # We use .pop() inside the lock to ensure that only one thread
        # can retrieve the data for a specific job_id.
        with self.lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id)

        # Step 2: Process the job outside the lock.
        # By doing this outside the lock, multiple threads can 
        # process different jobs simultaneously.
        try:
            result = Processor(data)
        except Exception as e:
            # Handle or log exception as needed
            print(f"Error processing job {job_id}: {e}")
            return False

        # Step 3: Atomically store the result.
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)