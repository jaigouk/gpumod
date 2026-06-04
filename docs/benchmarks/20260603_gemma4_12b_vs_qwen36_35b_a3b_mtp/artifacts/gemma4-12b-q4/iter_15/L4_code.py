import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to synchronize access to shared dictionaries
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        with self._lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # Step 1: Atomically check and "claim" the job
        with self._lock:
            if job_id not in self.jobs:
                return False

            # pop() removes the key and returns the value.
            # This prevents other threads from picking up the same job_id.
            data = self.jobs.pop(job_id)

        # Step 2: Run the processor outside of the lock.
        # This allows other threads to process other jobs concurrently.
        result = processor(data)

        # Step 3: Atomically write the result
        with self._lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self._lock:
            return self.results.get(job_id)

# Example usage:
if __name__ == "__main__":
    def mock_processor(data):
        return f"Processed: {data}"

    queue = JobQueue()
    queue.add_job("task1", "Some Data")

    # This will now safely execute in multi-threaded environments
    success = queue.process_job("task1", mock_processor)
    print(f"Success: {success}")
    print(f"Result: {queue.get_result('task1')}")