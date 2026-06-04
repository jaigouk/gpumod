import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # A single lock to protect access to both internal dictionaries
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        with self._lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # 1. Attempt to "claim" the job. 
        # We use a lock to ensure only one thread can extract a specific job.
        with self._lock:
            if job_id not in self.jobs:
                return False
            
            # pop() removes the job from the queue immediately so 
            # no other thread can attempt to process it.
            data = self.jobs.pop(job_id)

        # 2. Perform the heavy lifting OUTSIDE the lock.
        # This allows other threads to call add_job or process other jobs
        # while this thread is busy calculating.
        result = processor(data)

        # 3. Save the result.
        # We must re-acquire the lock to safely write to the results dictionary.
        with self._lock:
            self.results[job_id] = result
        
        return True

    def get_result(self, job_id):
        with self._lock:
            return self.results.get(job_id)

# --- Example Usage / Test ---
if __name__ == "__main__":
    import time

    def slow_processor(data):
        time.sleep(1)  # Simulate heavy work
        return data * 2

    queue = JobQueue()
    
    # Add a job
    queue.add_job(1, 10)

    def worker():
        success = queue.process_job(1, slow_processor)
        print(f"Worker {threading.current_thread().name} success: {success}")

    # Start two threads trying to process the SAME job
    t1 = threading.Thread(target=worker, name="A")
    t2 = threading.Thread(target=worker, name="B")

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    print(f"Final Result for Job 1: {queue.get_result(1)}")