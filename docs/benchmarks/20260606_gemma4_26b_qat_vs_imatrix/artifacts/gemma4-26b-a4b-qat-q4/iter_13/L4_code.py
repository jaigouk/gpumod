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
        # We use .pop() inside the lock to atomically retrieve AND remove the job.
        # This ensures that only one thread can "claim" a specific job_id.
        with self._lock:
            data = self.jobs.pop(job_id, None)

        # If data is None, the job_id didn't exist or was already picked up
        if data is None:
            return False

        # IMPORTANT: We run the processor OUTSIDE the lock.
        # If we kept the lock held here, no other thread could add or 
        # process jobs while this one is running, defeating the purpose 
        # of concurrency.
        result = processor(data)

        # Protect the write operation to the results dictionary
        with self._lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self._lock:
            return self.results.get(job_id)

# --- Example Usage / Test ---
if __name__ == "__main__":
    import time

    def heavy_computation(n):
        time.sleep(1)  # Simulate work
        return n * n

    queue = JobQueue()
    queue.add_job(1, 10)
    queue.add_job(1, 20) # Same ID to test race condition logic

    def worker(q, j_id, proc):
        success = q.process_job(j_id, proc)
        print(f"Job {j_id} processed: {success}")

    # Create multiple threads attempting to process the same job
    threads = []
    for _ in range(5):
        t = threading.Thread(target=worker, args=(queue, 1, heavy_computation))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print(f"Result: {queue.get_result(1)}")