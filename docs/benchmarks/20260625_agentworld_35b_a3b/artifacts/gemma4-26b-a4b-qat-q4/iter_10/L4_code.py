import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # A lock to protect access to the self.jobs and self.results dictionaries
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        with self._lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # 1. ATOMIC CLAIM: 
        # We use the lock to check if the job exists and remove it from 
        # the queue immediately so no other thread can claim it.
        with self._lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id)

        # 2. PARALLEL EXECUTION:
        # We run the processor OUTSIDE the lock. 
        # This allows other threads to process other jobs simultaneously.
        try:
            result = processor(data)
            processing_success = True
        except Exception as e:
            # Handle potential errors in the processor to prevent losing the result
            result = e
            processing_success = False

        # 3. ATOMIC WRITE:
        # Re-acquire the lock to safely store the result.
        with self._lock:
            self.results[job_id] = result

        return processing_success

    def get_result(self, job_id):
        with self._lock:
            return self.results.get(job_id)

# --- Testing the fix ---
if __name__ == "__main__":
    import time

    def heavy_work(n):
        time.sleep(1)  # Simulate long work
        return n * n

    queue = JobQueue()
    queue.add_job(1, 10)
    queue.add_job(1, 10)  # Duplicate ID to test race condition
    queue.add_job(2, 20)

    threads = []
    # Start multiple threads attempting to process the same and different jobs
    for i in range(5):
        t1 = threading.Thread(target=queue.process_job, args=(1, heavy_work))
        t2 = threading.Thread(target=queue.process_job, args=(2, heavy_work))
        threads.extend([t1, t2])
        t1.start()
        t2.start()

    for t in threads:
        t.join()

    print(f"Result for Job 1: {queue.get_result(1)}")
    print(f"Result for Job 2: {queue.get_result(2)}")