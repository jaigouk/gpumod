import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # A lock to synchronize access to the shared dictionaries
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        with self._lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # 1. Atomically check and remove the job from the queue
        # This prevents multiple threads from picking up the same job.
        with self._lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id)

        # 2. Perform the actual work OUTSIDE the lock.
        # This allows other threads to add jobs or process other jobs 
        # while this thread is busy computing.
        result = processor(data)

        # 3. Atomically store the result
        with self._lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self._lock:
            return self.results.get(job_id)

# --- Example Usage / Test Case ---
if __name__ == "__main__":
    import time

    def heavy_computation(n):
        time.sleep(1)  # Simulate a long-running task
        return n * n

    queue = JobQueue()

    # Add jobs
    for i in range(5):
        queue.add_job(i, i)

    # Define a worker function for threads
    def worker(job_id):
        success = queue.process_job(job_id, heavy_computation)
        if success:
            print(f"Job {job_id} completed.")
        else:
            print(f"Job {job_id} failed or already processed.")

    # Start multiple threads to process jobs simultaneously
    threads = []
    for i in range(5):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Verify results
    print("Results:", {i: queue.get_result(i) for i in range(5)})