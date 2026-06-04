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
        # 1. ATOMIC CHECK AND CLAIM
        # We use the lock to check if the job exists and remove it from the 
        # queue immediately. This "claims" the job so no other thread 
        # can pick it up.
        with self._lock:
            if job_id not in self.jobs:
                return False
            # .pop() removes and returns the value in one atomic step
            data = self.jobs.pop(job_id)

        # 2. PROCESSING (OUTSIDE THE LOCK)
        # We perform the heavy lifting outside the lock so other threads 
        # can add or claim different jobs simultaneously.
        try:
            result = processor(data)
            success = True
        except Exception as e:
            # In a real system, you might store the exception in results
            result = e
            success = False

        # 3. ATOMIC RESULT STORAGE
        # Re-acquire the lock to safely write the result.
        with self._lock:
            self.results[job_id] = result
        
        return success

    def get_result(self, job_id):
        with self._lock:
            return self.results.get(job_id)

# --- Testing the fix ---
if __name__ == "__main__":
    import time

    def heavy_processor(data):
        time.sleep(0.1)  # Simulate work
        return data * 2

    queue = JobQueue()
    queue.add_job(1, 10)
    queue.add_job(1, 10) # Adding same ID again for test

    def worker():
        # Attempt to process job 1
        success = queue.process_job(1, heavy_processor)
        print(f"Worker {threading.current_thread().name} success: {success}")

    # Create two threads trying to process the exact same job ID
    t1 = threading.Thread(target=worker, name="A")
    t2 = threading.Thread(target=worker, name="B")

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    print(f"Final Result for job 1: {queue.get_result(1)}")