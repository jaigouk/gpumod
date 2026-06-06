import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # A lock to synchronize access to the shared dictionaries
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        with self._lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # 1. Atomically check and remove the job from the queue
        # This ensures only one thread can successfully "claim" this job_id
        with self._lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id)

        # 2. Perform the heavy lifting OUTSIDE the lock
        # If we kept the lock during processor(data), no other jobs could 
        # be added or processed until this one finishes.
        result = processor(data)

        # 3. Atomically store the result
        with self._lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self._lock:
            return self.results.get(job_id)

# --- Example Usage/Test ---
if __name__ == "__main__":
    import time

    def slow_processor(data):
        time.sleep(1)  # Simulate heavy work
        return data * 2

    queue = JobQueue()

    # Add a job
    queue.add_job(1, 10)

    # Simulate multiple threads trying to process the same job
    def worker():
        success = queue.process_job(1, slow_processor)
        print(f"Thread {threading.current_thread().name} processing success: {success}")

    threads = [threading.Thread(target=worker, name=f"T{i}") for i in range(3)]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"Final Result: {queue.get_result(1)}")