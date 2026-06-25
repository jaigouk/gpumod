import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # A lock to ensure thread-safe access to the dictionaries
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # 1. Atomically check and remove the job from the queue.
        # This ensures only one thread can "claim" this job.
        with self.lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id)

        # 2. Run the processor OUTSIDE the lock.
        # If we ran the processor inside the lock, we would block 
        # all other threads from adding or getting jobs until this 
        # heavy computation finishes.
        result = processor(data)

        # 3. Atomically store the result.
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)

# --- Example Usage / Testing ---
if __name__ == "__main__":
    import time

    def heavy_computation(n):
        time.sleep(0.1)  # Simulate work
        return n * n

    queue = JobQueue()
    queue.add_job(1, 10)
    queue.add_job(2, 20)

    def worker(job_id):
        success = queue.process_job(job_id, heavy_computation)
        print(f"Job {job_id} processed: {success}")

    threads = []
    # Simulate multiple threads trying to process the same job ID
    for _ in range(3):
        threads.append(threading.Thread(target=worker, args=(1,)))
        threads.append(threading.Thread(target=worker, args=(2,)))

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    print(f"Results: {queue.get_result(1)}, {queue.get_result(2)}")