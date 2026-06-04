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
        data = None

        # Step 1: Atomically "claim" the job by removing it from the jobs dict
        with self.lock:
            # .pop() returns the value if it exists, otherwise None
            # This prevents two threads from processing the same ID simultaneously
            data = self.jobs.pop(job_id, None)

        if data is None:
            return False

        # Step 2: Perform the actual work outside of the lock.
        # This allows multiple threads to run different processors concurrently.
        result = Processor(data)

        # Step 3: Store the result safely
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)

# Example Usage:
if __name__ == "__main__":
    queue = JobQueue()
    queue.add_job("job1", {"task": "compute_pi"})

    def mock_processor(data):
        return f"Processed {data['task']}"

    success = queue.process_job("job1", mock_processor)
    print(f"Success: {success}")
    print(f"Result: {queue.get_result('job1')}")