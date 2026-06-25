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
        # 1. "Claim" the job: Atomically check if it exists and remove it 
        # from the pending jobs dictionary. This prevents other threads 
        # from picking up the same job.
        with self.lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id)

        # 2. Execute the processor outside the lock.
        # This is critical: it allows other threads to call add_job() 
        # or process other jobs while this one is running.
        try:
            result = processor(data)
        except Exception as e:
            # In a real system, you might store the exception in self.results
            raise e

        # 3. Store the result atomically
        with self.lock:
            self.results[job_id] = result
        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)

# --- Example Usage / Test ---
if __name__ == "__main__":
    import time

    def slow_processor(data):
        time.sleep(1)  # Simulate heavy work
        return data * 2

    queue = JobQueue()
    queue.add_job(1, 10)
    queue.add_job(2, 20)

    # Launch multiple threads trying to process the same job
    threads = []
    for _ in range(5):
        t = threading.Thread(target=queue.process_job, args=(1, slow_processor))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print(f"Result for Job 1: {queue.get_result(1)}") # Should be 20
    print(f"Result for Job 2: {queue.get_result(2)}") # Should be 40