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
        # 1. Atomically check for existence and remove the job from the queue.
        # This ensures that even if multiple threads call process_job with 
        # the same ID, only one thread will successfully 'pop' the data.
        with self.lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id)

        # 2. Run the processor OUTSIDE the lock.
        # This allows other threads to call add_job, get_result, or 
        # process other jobs while this one is computationally expensive.
        result = processor(data)

        # 3. Atomically store the result.
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
        time.sleep(0.1)  # Simulate work
        return data * 2

    queue = JobQueue()

    # Add a job
    queue.add_job("task1", 10)

    # Simulate multiple threads trying to process the same job simultaneously
    def worker():
        success = queue.process_job("task1", slow_processor)
        print(f"Thread {threading.current_thread().name} success: {success}")

    threads = [threading.Thread(target=worker, name=f"T{i}") for i in range(5)]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"Final Result: {queue.get_result('task1')}")