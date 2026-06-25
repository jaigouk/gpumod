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
        # 1. Atomically check if the job exists and remove it from the queue.
        # This ensures only one thread can ever "own" a specific job_id.
        with self._lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id)

        # 2. Execute the processor OUTSIDE the lock.
        # This allows other threads to add jobs or process other jobs 
        # while this thread is doing the heavy lifting.
        result = processor(data)

        # 3. Atomically store the result.
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
    queue.add_job(1, 10)
    queue.add_job(1, 10)  # Attempting to add same ID
    queue.add_job(2, 20)

    def worker(job_id):
        success = queue.process_job(job_id, slow_processor)
        print(f"Job {job_id} processed: {success}")

    # Create multiple threads to try and process the same jobs
    threads = [
        threading.Thread(target=worker, args=(1,)),
        threading.Thread(target=worker, args=(1,)), # This should return False immediately
        threading.Thread(target=worker, args=(2,)),
    ]

    for t in threads: t.start()
    for t in threads: t.join()

    print(f"Results: {queue.get_result(1)}")
    print(f"Results: {queue.get_result(2)}")