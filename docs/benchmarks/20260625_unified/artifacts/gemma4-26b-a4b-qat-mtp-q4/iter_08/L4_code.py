import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # A single lock to protect access to the shared dictionaries
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # 1. ATOMIC TAKE:
        # We lock only to check existence and remove the job from the queue.
        # This prevents multiple threads from processing the same job_id.
        with self.lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id)

        # 2. WORK EXECUTION:
        # We perform the heavy lifting OUTSIDE the lock.
        # This allows other threads to call add_job or process_job concurrently.
        try:
            result = processor(data)
            processing_successful = True
        except Exception as e:
            result = e
            processing_successful = False

        # 3. ATOMIC WRITE:
        # We lock again only to write the result to the results dictionary.
        with self.lock:
            self.results[job_id] = result

        return processing_successful

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)

# --- Example Usage/Test ---
if __name__ == "__main__":
    import time

    def slow_processor(data):
        time.sleep(1)  # Simulate heavy work
        return data * 2

    queue = JobQueue()
    queue.add_job(1, 10)
    queue.add_job(2, 20)

    def worker():
        queue.process_job(1, slow_processor)
        queue.process_job(2, slow_processor)

    threads = [threading.Thread(target=worker) for _ in range(3)]

    for t in threads: t.start()
    for t in threads: t.join()

    print(f"Result 1: {queue.get_result(1)}") # Expected: 20
    print(f"Result 2: {queue.get_result(2)}") # Expected: 40