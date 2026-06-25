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
        # 1. Atomic Check and Retrieval
        # We use the lock to check and remove the job from the queue simultaneously.
        # This prevents multiple threads from picking up the same job.
        with self.lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id)

        # 2. Execution (OUTSIDE the lock)
        # We call the processor outside the lock so that other threads 
        # can continue to add or process other jobs while this one runs.
        result = processor(data)

        # 3. Atomic Result Writing
        # Once the work is done, we re-acquire the lock to save the result.
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)

# --- Example Usage / Test ---
if __name__ == "__main__":
    import time

    queue = JobQueue()

    def slow_processor(data):
        time.sleep(1)  # Simulate heavy work
        return data * 2

    def worker(job_id, data):
        print(f"Starting job {job_id}")
        success = queue.process_job(job_id, slow_processor)
        if success:
            print(f"Finished job {job_id}")

    # Create multiple threads to attempt to process the same job
    # Only one should succeed; the others should return False.
    queue.add_job(1, 10)

    threads = []
    for _ in range(5):
        t = threading.Thread(target=worker, args=(1, None))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print(f"Final Result for job 1: {queue.get_result(1)}")