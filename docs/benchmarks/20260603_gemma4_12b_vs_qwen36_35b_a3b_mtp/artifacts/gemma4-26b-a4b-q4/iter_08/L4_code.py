import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # A lock to ensure atomicity when accessing shared dictionaries
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        with self._lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # 1. Critical Section: Check existence and extract data
        # We wrap this in a lock to prevent two threads from picking up the same job.
        with self._lock:
            if job_id not in self.jobs:
                return False
            
            # Retrieve data and immediately remove it from the jobs dict
            # so no other thread can attempt to process it.
            data = self.jobs.pop(job_id)

        # 2. Execution: Perform the heavy lifting OUTSIDE the lock.
        # This allows other threads to add_job or process other jobs 
        # while this specific job is being computed.
        result = processor(data)

        # 3. Critical Section: Save the result
        with self._lock:
            self.results[job_id] = result
            
        return True

    def get_result(self, job_id):
        with self._lock:
            return self.results.get(job_id)

# --- Demonstration of Thread Safety ---
if __name__ == "__main__":
    import time

    def slow_processor(data):
        time.sleep(0.1)  # Simulate heavy work
        return data * 2

    queue = JobQueue()
    queue.add_job(1, 10)
    queue.add_job(1, 10) # Attempting to add same ID (re-submission)

    def worker(job_id):
        success = queue.process_job(job_id, slow_processor)
        print(f"Worker for job {job_id} success: {success}")

    # Create multiple threads trying to process the SAME job ID
    threads = [threading.Thread(target=worker, args=(1,)) for _ in range(5)]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"Final Result for job 1: {queue.get_result(1)}")