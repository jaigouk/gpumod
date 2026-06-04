import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to protect access to shared dictionaries
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        with self._lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # 1. Safely extract the job data from the dictionary
        # We use the lock to perform a "check-and-remove" operation atomically
        with self._lock:
            if job_id not in self.jobs:
                return False
            # pop() removes the item and returns it, ensuring no other 
            # thread can pick up this specific job_id
            data = self.jobs.pop(job_id)

        # 2. Execute the processor OUTSIDE the lock.
        # This allows other threads to add/process other jobs while this one runs.
        result = processor(data)

        # 3. Safely store the result
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
        time.sleep(0.1)  # Simulate heavy work
        return data * 2

    queue = JobQueue()

    # Add jobs
    queue.add_job(1, 10)
    queue.add_job(2, 20)
    queue.add_job(3, 30)

    def worker(job_id):
        success = queue.process_job(job_id, slow_processor)
        print(f"Job {job_id} processed: {success}")

    # Start multiple threads to process jobs simultaneously
    threads = []
    for i in range(1, 4):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Check results
    print(f"Result 1: {queue.get_result(1)}")
    print(f"Result 2: {queue.get_result(2)}")
    print(f"Result 3: {queue.get_result(3)}")