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
        # 1. Atomically check if job exists and "claim" it by removing it
        # This prevents multiple threads from processing the same job_id.
        with self._lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id)

        # 2. Run the processor OUTSIDE the lock.
        # This allows other threads to add/get jobs while this one is computing.
        try:
            result = processor(data)
            result_success = True
        except Exception as e:
            result = e
            result_success = False

        # 3. Atomically store the result
        with self._lock:
            self.results[job_id] = result

        return result_success

    def get_result(self, job_id):
        with self._lock:
            return self.results.get(job_id)

# --- Example Usage ---
if __name__ == "__main__":
    import time

    def heavy_computation(n):
        time.sleep(1)  # Simulate work
        return n * n

    queue = JobQueue()
    queue.add_job(1, 10)
    queue.add_job(2, 20)

    def worker(job_id):
        if queue.process_job(job_id, heavy_computation):
            print(f"Job {job_id} finished.")
        else:
            print(f"Job {job_id} failed or already processed.")

    threads = [
        threading.Thread(target=worker, args=(1,)),
        threading.Thread(target=worker, args=(1,)), # Attempting to process same job
        threading.Thread(target=worker, args=(2,)),
    ]

    for t in threads: t.start()
    for t in threads: t.join()

    print(f"Results: {queue.results}")