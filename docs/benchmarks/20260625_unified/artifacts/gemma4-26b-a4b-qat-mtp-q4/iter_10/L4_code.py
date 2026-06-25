import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to ensure thread safety
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        with self._lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # 1. Atomically "claim" the job by popping it from the dictionary.
        # This prevents multiple threads from processing the same job_id.
        with self._lock:
            data = self.jobs.pop(job_id, None)

        # If data is None, the job_id wasn't in the queue (or was already claimed)
        if data is None:
            return False

        # 2. Perform the processing OUTSIDE the lock.
        # This allows other threads to add/get jobs while this one works.
        result = processor(data)

        # 3. Re-acquire the lock to save the result.
        with self._lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self._lock:
            return self.results.get(job_id)

# --- Testing the fix ---
if __name__ == "__main__":
    import time

    def slow_processor(data):
        time.sleep(0.1)  # Simulate heavy work
        return data * 2

    queue = JobQueue()
    queue.add_job(1, 10)
    queue.add_job(1, 10)  # Attempting to add the same ID again

    # Create multiple threads trying to process the same job
    threads = []
    for _ in range(5):
        t = threading.Thread(target=queue.process_job, args=(1, slow_processor))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print(f"Result: {queue.get_result(1)}") # Expected: 20