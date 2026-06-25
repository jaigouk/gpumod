import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # A lock to ensure all dictionary operations are thread-safe
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        with self._lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # 1. Attempt to "claim" the job by popping it from the jobs dict.
        # We hold the lock during the pop to ensure only one thread gets the data.
        with self._lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id)

        # 2. Perform the actual processing.
        # IMPORTANT: We do NOT hold the lock here. If we did, the entire 
        # job queue would become single-threaded, defeating the purpose.
        try:
            result = processor(data)
            processing_successful = True
        except Exception:
            processing_successful = False
            result = None

        # 3. Store the result.
        # We re-acquire the lock to safely write to the results dictionary.
        with self._lock:
            if processing_successful:
                self.results[job_id] = result
            return True

    def get_result(self, job_id):
        with self._lock:
            return self.results.get(job_id)

# --- Example Usage / Test ---
if __name__ == "__main__":
    import time

    def heavy_task(n):
        time.sleep(1)  # Simulate work
        return n * n

    queue = JobQueue()
    queue.add_job(1, 10)
    queue.add_job(2, 20)

    # Use threads to process jobs concurrently
    t1 = threading.Thread(target=queue.process_job, args=(1, heavy_task))
    t2 = threading.Thread(target=queue.process_job, args=(2, heavy_task))
    t3 = threading.Thread(target=queue.process_job, args=(1, heavy_task)) # Duplicate attempt

    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()

    print(f"Result 1: {queue.get_result(1)}") # Expected: 100
    print(f"Result 2: {queue.get_result(2)}") # Expected: 400