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
        # 1. ATOMIC CLAIM: Check if job exists and remove it from 'jobs' 
        # so no other thread can pick it up.
        data = None
        with self._lock:
            if job_id in self.jobs:
                # pop() removes the item and returns it, ensuring 
                # only one thread can ever get this data.
                data = self.jobs.pop(job_id)
            else:
                return False

        # 2. EXECUTION: Run the processor OUTSIDE the lock.
        # This allows other threads to add/remove jobs while this one works.
        result = processor(data)

        # 3. ATOMIC STORE: Save the result back to the shared dictionary.
        with self._lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self._lock:
            return self.results.get(job_id)

# --- Testing the implementation ---
if __name__ == "__main__":
    import time

    def heavy_computation(n):
        time.sleep(0.1)  # Simulate work
        return n * n

    queue = JobQueue()

    def worker(job_id, val):
        queue.add_job(job_id, val)
        # Try to process
        queue.process_job(job_id, heavy_computation)

    # Launch multiple threads trying to process the same job_id
    # to demonstrate that only one succeeds in "claiming" it.
    threads = []
    for i in range(5):
        # All threads trying to claim job '100'
        t = threading.Thread(target=worker, args=("100", 10))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print(f"Result: {queue.get_result('100')}") # Should be 100