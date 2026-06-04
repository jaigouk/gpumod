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
        # 1. ATOMIC CHECK AND REMOVE
        # We use the lock to check if the job exists and remove it 
        # immediately so no other thread can pick up the same job.
        with self._lock:
            if job_id not in self.jobs:
                return False
            
            # .pop() removes the item and returns it in one atomic-like step 
            # within our lock, preventing other threads from seeing it.
            data = self.jobs.pop(job_id)

        # 2. PROCESSING (OUTSIDE THE LOCK)
        # We execute the processor outside the lock. This allows other 
        # threads to call add_job or process_job for DIFFERENT IDs 
        # while this one is working.
        result = processor(data)

        # 3. ATOMIC RESULT STORAGE
        # We re-acquire the lock only to write the result.
        with self._lock:
            self.results[job_id] = result
            
        return True

    def get_result(self, job_id):
        with self._lock:
            return self.results.get(job_id)

# --- Testing the fix ---
if __name__ == "__main__":
    import time

    def slow_processor(data):
        time.sleep(0.1)  # Simulate heavy work
        return data * 2

    queue = JobQueue()
    queue.add_job(1, 10)
    queue.add_job(1, 10) # Attempting to add same ID is fine, but process_job logic handles it

    def worker(job_id):
        success = queue.process_job(job_id, slow_processor)
        print(f"Job {job_id} processed: {success}")

    # Create two threads attempting to process the same job ID
    t1 = threading.Thread(target=worker, args=(1,))
    t2 = threading.Thread(target=worker, args=(1,))

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    print(f"Result for job 1: {queue.get_result(1)}")