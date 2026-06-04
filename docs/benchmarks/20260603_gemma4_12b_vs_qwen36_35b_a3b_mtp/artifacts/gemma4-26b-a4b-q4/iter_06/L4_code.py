import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # A lock to protect access to shared state (jobs and results)
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        with self._lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # 1. Atomically check and "claim" the job
        # We use pop() inside the lock so that only ONE thread can 
        # ever retrieve this specific job_id.
        with self._lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id)

        # 2. Process the job OUTSIDE the lock
        # This allows other threads to add jobs or process other jobs 
        # while this one is running.
        try:
            result = processor(data)
            job_success = True
        except Exception as e:
            result = e
            job_success = False

        # 3. Atomically write the result
        with self._lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self._lock:
            return self.results.get(job_id)

# --- Example Usage / Test ---
if __name__ == "__main__":
    import time

    def heavy_task(n):
        time.sleep(1)  # Simulate work
        return n * n

    queue = JobQueue()

    # Add some jobs
    queue.add_job(1, 10)
    queue.add_job(2, 20)
    queue.add_job(3, 30)

    def worker(job_id):
        print(f"Starting job {job_id}")
        if queue.process_job(job_id, heavy_task):
            print(f"Finished job {job_id}")

    # Start multiple threads to process jobs simultaneously
    threads = []
    for i in range(1, 4):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print("Results:", queue.get_result(1), queue.get_result(2), queue.get_result(3))