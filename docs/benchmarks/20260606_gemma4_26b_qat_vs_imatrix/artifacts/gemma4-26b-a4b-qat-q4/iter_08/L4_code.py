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
        # 1. Atomic "Claim" phase
        # We use pop() inside the lock to retrieve and remove the job in one step.
        # This ensures only ONE thread can ever "own" this job_id for processing.
        with self._lock:
            data = self.jobs.pop(job_id, None)

        if data is None:
            return False

        # 2. Processing phase
        # We perform the actual work OUTSIDE the lock. 
        # This allows other threads to call add_job or process other jobs concurrently.
        result = processor(data)

        # 3. Result storage phase
        # Re-acquire the lock to safely write the result.
        with self._lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self._lock:
            return self.results.get(job_id)

# --- Example Usage / Test Case ---
if __name__ == "__main__":
    import time

    queue = JobQueue()

    def slow_processor(data):
        time.sleep(0.1)  # Simulate heavy work
        return data * 2

    # Add a job
    queue.add_job("job1", 10)

    # Simulate multiple threads trying to process the same job
    def worker(job_id):
        success = queue.process_job(job_id, slow_processor)
        print(f"Thread {threading.current_thread().name} success: {success}")

    threads = []
    for i in range(5):
        t = threading.Thread(target=worker, args=("job1",), name=f"T{i}")
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print(f"Final Result: {queue.get_result('job1')}")