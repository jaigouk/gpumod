import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # A lock to protect access to shared dictionaries
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        with self._lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # 1. Atomically retrieve and remove the job from the queue.
        # This prevents multiple threads from picking up the same job.
        with self._lock:
            data = self.jobs.pop(job_id, None)

        # If data is None, the job_id was not in the dictionary 
        # (either it never existed or another thread already popped it).
        if data is None:
            return False

        # 2. Execute the processor OUTSIDE the lock.
        # If we held the lock here, the entire queue would block until 
        # this specific job finished, destroying concurrency.
        result = processor(data)

        # 3. Safely store the result.
        with self._lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self._lock:
            return self.results.get(job_id)

# --- Example Usage / Test ---
if __name__ == "__main__":
    import time

    def heavy_task(data):
        time.sleep(1)  # Simulate work
        return data * 2

    queue = JobQueue()
    queue.add_job(1, 10)
    queue.add_job(2, 20)

    # Simulate multiple threads trying to process the same job IDs
    def worker(qid):
        success = queue.process_job(qid, heavy_task)
        print(f"Thread processing job {qid}: {'Success' if success else 'Failed/Already Taken'}")

    threads = []
    # Creating 4 threads to fight over 2 jobs
    for i in range(4):
        t = threading.Thread(target=worker, args=(i % 2 + 1,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print(f"Result for Job 1: {queue.get_result(1)}")
    print(f"Result for Job 2: {queue.get_result(2)}")