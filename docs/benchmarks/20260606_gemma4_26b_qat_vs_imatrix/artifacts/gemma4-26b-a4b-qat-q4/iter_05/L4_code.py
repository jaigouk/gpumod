import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # A single lock to protect access to the shared dictionaries
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        with self._lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # 1. ATOMIC CLAIM: Check if job exists and remove it from pending 
        # tasks immediately so no other thread can claim it.
        with self._lock:
            if job_id not in self.jobs:
                return False
            # pop() removes the item and returns it in one atomic step
            data = self.jobs.pop(job_id)

        # 2. PROCESSING: Perform the work OUTSIDE the lock.
        # This allows other threads to add/process other jobs simultaneously.
        result = processor(data)

        # 3. STORE RESULT: Re-acquire the lock to write the result.
        with self._lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self._lock:
            return self.results.get(job_id)

# --- Example Usage/Test ---
if __name__ == "__main__":
    import time

    queue = JobQueue()

    def heavy_work(n):
        time.sleep(1)  # Simulate long task
        return n * n

    # Add jobs
    queue.add_job(1, 10)
    queue.add_job(2, 20)

    # Simulate multiple threads trying to process the SAME job ID
    # One should succeed (return True), the other should fail (return False)
    def worker(job_id):
        success = queue.process_job(job_id, heavy_work)
        print(f"Job {job_id} processed successfully: {success}")

    t1 = threading.Thread(target=worker, args=(1,))
    t2 = threading.Thread(target=worker, args=(1,))
    t3 = threading.Thread(target=worker, args=(2,))

    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()

    print(f"Result for Job 1: {queue.get_result(1)}")
    print(f"Result for Job 2: {queue.get_result(2)}")