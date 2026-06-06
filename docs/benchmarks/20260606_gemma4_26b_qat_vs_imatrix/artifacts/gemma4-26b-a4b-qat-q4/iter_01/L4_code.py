import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # A lock to protect access to the shared dictionaries
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        with self._lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # 1. Atomically check if job exists and remove it from the queue.
        # This prevents two threads from processing the same job_id.
        with self._lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id)

        # 2. Execute the processor OUTSIDE the lock.
        # This allows other threads to add jobs or get results while 
        # this thread is performing heavy computation/IO.
        result = processor(data)

        # 3. Re-acquire the lock to safely write the result.
        with self._lock:
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
        return n * 2

    queue = JobQueue()
    queue.add_job(1, 10)
    queue.add_job(2, 20)

    def worker(q, j_id):
        print(f"Thread starting job {j_id}")
        if q.process_job(j_id, heavy_task):
            print(f"Thread finished job {j_id}")

    threads = [
        threading.Thread(target=worker, args=(queue, 1)),
        threading.Thread(target=worker, args=(queue, 2)),
        # Attempting to process job 1 again should return False immediately
        threading.Thread(target=worker, args=(queue, 1)), 
    ]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    print(f"Results: {queue.get_result(1)}")
    print(f"Results: {queue.get_result(2)}")