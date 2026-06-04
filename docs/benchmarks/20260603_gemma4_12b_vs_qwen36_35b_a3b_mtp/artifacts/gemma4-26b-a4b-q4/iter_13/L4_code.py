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
        # 1. Check and Claim Phase
        with self._lock:
            if job_id not in self.jobs:
                return False

            # We pop the job immediately. This ensures that if another thread
            # tries to process the same job_id, it won't find it in self.jobs.
            data = self.jobs.pop(job_id)

        # 2. Processing Phase
        # We perform the heavy lifting OUTSIDE the lock.
        # This allows multiple threads to run different processors in parallel.
        result = processor(data)

        # 3. Result Storage Phase
        with self._lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self._lock:
            return self.results.get(job_id)

# --- Example Usage/Test ---
if __name__ == "__main__":
    import time

    def heavy_task(n):
        time.sleep(1)  # Simulate work
        return n * n

    queue = JobQueue()

    # Add some jobs
    queue.add_job(1, 10)
    queue.add_job(2, 20)
    queue.add_job(3, 30)

    def worker(job_id):
        print(f"Thread starting job {job_id}")
        if queue.process_job(job_id, heavy_task):
            print(f"Thread finished job {job_id}")
        else:
            print(f"Thread failed to find job {job_id}")

    # Start multiple threads to process jobs concurrently
    threads = []
    for i in [1, 2, 3]:
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print("Results:", queue.get_result(1), queue.get_result(2), queue.get_result(3))