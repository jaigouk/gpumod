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
        # 1. Atomically check if job exists and "claim" it by removing it from self.jobs
        with self._lock:
            if job_id not in self.jobs:
                return False
            # pop() removes the item and returns it, preventing other threads 
            # from seeing this job_id in self.jobs
            data = self.jobs.pop(job_id)

        # 2. Execute the processor OUTSIDE the lock. 
        # This allows other threads to add or process other jobs concurrently.
        try:
            result = processor(data)
            job_success = True
        except Exception as e:
            result = e
            job_success = False

        # 3. Re-acquire the lock to safely write the result
        with self._lock:
            self.results[job_id] = result
            
        return job_success

    def get_result(self, job_id):
        with self._lock:
            return self.results.get(job_id)

# --- Example Usage/Test ---
if __name__ == "__main__":
    import time

    def slow_processor(data):
        time.sleep(1)  # Simulate heavy work
        return data * 2

    queue = JobQueue()
    
    # Add jobs
    queue.add_job(1, 10)
    queue.add_job(2, 20)

    def worker(job_id):
        print(f"Starting job {job_id}")
        queue.process_job(job_id, slow_processor)
        print(f"Finished job {job_id}")

    # Start multiple threads to process jobs
    t1 = threading.Thread(target=worker, args=(1,))
    t2 = threading.Thread(target=worker, args=(2,))
    t3 = threading.Thread(target=worker, args=(1,)) # Attempting to process same job twice

    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()

    print(f"Result for job 1: {queue.get_result(1)}")
    print(f"Result for job 2: {queue.get_result(2)}")