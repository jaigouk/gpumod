import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # A lock to protect access to the internal dictionaries
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        with self._lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # Step 1: Atomically check for existence and "claim" the job
        # We use pop() inside the lock to ensure only one thread 
        # can ever retrieve the data for a specific job_id.
        with self._lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id)

        # Step 2: Perform the actual processing 
        # We do this OUTSIDE the lock so other threads can 
        # add/process other jobs while this one runs.
        result = processor(data)

        # Step 3: Atomically save the result
        with self._lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self._lock:
            return self.results.get(job_id)

# --- Example Usage / Test ---
if __name__ == "__main__":
    import time

    def slow_processor(data):
        time.sleep(0.1)  # Simulate work
        return data * 2

    queue = JobQueue()
    queue.add_job(1, 10)
    queue.add_job(1, 20)  # Adding same ID again to test race condition

    threads = []
    for _ in range(5):
        # Multiple threads trying to process the same job_id (1)
        t = threading.Thread(target=queue.process_job, args=(1, slow_processor))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print(f"Result: {queue.get_result(1)}")