import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # A single lock to protect access to the internal dictionaries
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # 1. Atomically check and remove the job from the queue.
        # This ensures only ONE thread can ever "own" this job.
        with self.lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id)

        # 2. Execute the processor OUTSIDE the lock.
        # This allows other threads to add/process/get results while this runs.
        result = processor(data)

        # 3. Atomically store the result.
        with self.lock:
            self.results[job_id] = result
        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)

# --- Example Usage / Testing ---
if __name__ == "__main__":
    import time

    def slow_processor(data):
        time.sleep(0.1)  # Simulate work
        return data * 2

    queue = JobQueue()

    # Add a job
    queue.add_job("task1", 10)

    # Simulate multiple threads trying to process the same job
    def worker():
        success = queue.process_job("task1", slow_processor)
        print(f"Worker {threading.current_thread().name} success: {success}")

    threads = [threading.Thread(target=worker, name=f"T{i}") for i in range(3)]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"Final Result: {queue.get_result('task1')}")