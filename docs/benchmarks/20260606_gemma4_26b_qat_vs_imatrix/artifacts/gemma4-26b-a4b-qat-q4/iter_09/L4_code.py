import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # A lock to protect access to the shared dictionaries
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # 1. ATOMIC CLAIM: We use the lock to check if the job exists 
        # and remove it from the pending jobs immediately.
        # This "claims" the job so no other thread can process it.
        with self.lock:
            data = self.jobs.pop(job_id, None)

        # If data is None, the job_id wasn't in self.jobs (or was already claimed)
        if data is None:
            return False

        # 2. EXECUTION: Run the processor OUTSIDE the lock.
        # This allows other threads to call add_job or process_job 
        # while this thread is doing the heavy lifting.
        result = processor(data)

        # 3. ATOMIC WRITE: Re-acquire the lock to save the result.
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)

# --- Example Usage/Test ---
if __name__ == "__main__":
    import time

    queue = JobQueue()

    def heavy_task(n):
        time.sleep(1)  # Simulate long work
        return n * n

    # Add a job
    queue.add_job(1, 10)

    # Simulate multiple threads trying to process the same job
    def worker():
        success = queue.process_job(1, heavy_task)
        print(f"Thread {threading.current_thread().name} success: {success}")

    threads = [threading.Thread(target=worker, name=f"T{i}") for i in range(3)]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    print(f"Final Result: {queue.get_result(1)}")