import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # A lock to ensure thread-safe access to our dictionaries
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        with self._lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # 1. Attempt to "claim" the job atomically
        with self._lock:
            # .pop() removes the item and returns it. 
            # If job_id isn't there, it returns None.
            data = self.jobs.pop(job_id, None)

        if data is None:
            return False

        # 2. Execute the processor OUTSIDE the lock.
        # This is crucial so other threads aren't blocked while this runs.
        result = processor(data)

        # 3. Store the result safely
        with self._lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self._lock:
            return self.results.get(job_id)

# --- Testing the fix ---
if __name__ == "__main__":
    import time

    queue = JobQueue()

    def heavy_task(n):
        time.sleep(1)  # Simulate long work
        return n * 2

    # Add a job
    queue.add_job("task1", 21)

    # Simulate multiple threads trying to process the same job simultaneously
    def worker():
        success = queue.process_job("task1", heavy_task)
        print(f"Worker success: {success}")

    threads = [threading.Thread(target=worker) for _ in range(5)]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"Final Result: {queue.get_result('task1')}")