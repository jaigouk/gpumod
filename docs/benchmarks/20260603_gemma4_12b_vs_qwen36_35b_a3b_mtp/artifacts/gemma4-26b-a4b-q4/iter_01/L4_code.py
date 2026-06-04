import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        with self._lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # 1. Attempt to "claim" the job atomically
        with self._lock:
            if job_id not in self.jobs:
                return False
            # pop() removes the item and returns it, preventing other 
            # threads from seeing this job_id in self.jobs
            data = self.jobs.pop(job_id)

        # 2. Execute the processor OUTSIDE the lock.
        # This allows other threads to process other jobs concurrently.
        try:
            result = processor(data)
            processing_successful = True
        except Exception as e:
            result = e
            processing_successful = False

        # 3. Write the result back to the dictionary safely
        with self._lock:
            self.results[job_id] = result
            
        return True

    def get_result(self, job_id):
        with self._lock:
            return self.results.get(job_id)

# --- Verification Example ---
if __name__ == "__main__":
    import time

    queue = JobQueue()

    def slow_processor(data):
        time.sleep(1)  # Simulate heavy work
        return data * 2

    # Add a job
    queue.add_job(1, 10)

    # Simulate two threads trying to process the same job ID simultaneously
    def worker():
        success = queue.process_job(1, slow_processor)
        print(f"Thread {threading.current_thread().name} success: {success}")

    t1 = threading.Thread(target=worker, name="A")
    t2 = threading.Thread(target=worker, name="B")

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    print(f"Final Result: {queue.get_result(1)}")