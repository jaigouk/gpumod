import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # A single lock to protect access to the internal dictionaries
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        with self._lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # 1. Atomically "claim" the job.
        # We use pop() inside the lock so that only one thread can 
        # successfully retrieve the data for a specific job_id.
        with self._lock:
            data = self.jobs.pop(job_id, None)

        # If data is None, the job didn't exist or was already claimed
        if data is None:
            return False

        # 2. Execute the processor OUTSIDE the lock.
        # This allows other threads to add or process other jobs 
        # while this thread is doing heavy lifting.
        result = processor(data)

        # 3. Atomically store the result.
        with self._lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self._lock:
            return self.results.get(job_id)

# --- Example Usage / Testing ---
if __name__ == "__main__":
    import time

    def slow_processor(data):
        time.sleep(0.1)  # Simulate heavy work
        return data * 2

    queue = JobQueue()
    queue.add_job(1, 10)
    queue.add_job(2, 20)

    def worker(job_id):
        success = queue.process_job(job_id, slow_processor)
        print(f"Job {job_id} processed: {success}")

    # Start multiple threads trying to process the SAME job
    # Only one should succeed (return True)
    threads = []
    for _ in range(5):
        t1 = threading.Thread(target=worker, args=(1,))
        t2 = threading.Thread(target=worker, args=(2,))
        threads.extend([t1, t2])

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"Result for Job 1: {queue.get_result(1)}")
    print(f"Result for Job 2: {queue.get_result(2)}")