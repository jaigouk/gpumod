import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # A lock to protect access to the jobs and results dictionaries
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
        
        # If data is None, the job_id didn't exist or was already claimed/processed
        if data is None:
            return False

        # 2. Perform the heavy lifting (processing) OUTSIDE the lock.
        # This allows other threads to add/process different jobs in parallel.
        result = processor(data)

        # 3. Atomically store the result.
        with self._lock:
            self.results[job_id] = result
            
        return True

    def get_result(self, job_id):
        with self._lock:
            return self.results.get(job_id)

# --- Testing the implementation ---
if __name__ == "__main__":
    import time

    def slow_processor(data):
        time.sleep(0.1)  # Simulate work
        return data * 2

    queue = JobQueue()
    queue.add_job(1, 10)
    queue.add_job(2, 20)

    def worker(job_id):
        success = queue.process_job(job_id, slow_processor)
        print(f"Thread for job {job_id}: {'Success' if success else 'Failed/Already Taken'}")

    # Try to process the same job twice simultaneously using multiple threads
    t1 = threading.Thread(target=worker, args=(1,))
    t2 = threading.Thread(target=worker, args=(1,)) # Duplicate job attempt
    t3 = threading.Thread(target=worker, args=(2,)) # Different job

    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()

    print(f"Result for job 1: {queue.get_result(1)}")
    print(f"Result for job 2: {queue.get_result(2)}")