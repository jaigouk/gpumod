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
        # Step 1: Claim the job safely
        # We use a lock to ensure only one thread can pop a specific job_id
        with self._lock:
            if job_id not in self.jobs:
                return False
            # .pop() removes the item and returns it in one atomic step
            data = self.jobs.pop(job_id)

        # Step 2: Process the job 
        # We do NOT hold the lock here. This allows other threads to 
        # add/process other jobs while this one is running.
        try:
            result = processor(data)
        except Exception as e:
            # Optional: Handle processing errors (e.g., log them)
            print(f"Error processing job {job_id}: {e}")
            raise e

        # Step 3: Store the result
        with self._lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self._lock:
            return self.results.get(job_id)

# Example Usage:
if __name__ == "__main__":
    import time

    queue = JobQueue()
    queue.add_job("task1", {"value": 10})

    def slow_processor(data):
        time.sleep(1) # Simulate work
        return data["value"] * 2

    # This would have caused issues in the original code if called 
    # by multiple threads simultaneously.
    success = queue.process_job("task1", slow_processor)
    print(f"Success: {success}, Result: {queue.get_result('task1')}")