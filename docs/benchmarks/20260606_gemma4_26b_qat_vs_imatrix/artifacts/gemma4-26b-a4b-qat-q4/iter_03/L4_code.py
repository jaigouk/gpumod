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
        # 1. Atomic "Check and Take" phase
        # We lock only to check existence and remove the job from the queue.
        # This ensures only ONE thread ever retrieves a specific job_id.
        with self._lock:
            if job_id not in self.jobs:
                return False
            # .pop() removes the job so no other thread can grab it
            data = self.jobs.pop(job_id)

        # 2. Execution phase
        # We do NOT hold the lock while running the processor.
        # This allows multiple threads to run different jobs in parallel.
        result = processor(data)

        # 3. Atomic "Store Result" phase
        with self._lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self._lock:
            return self.results.get(job_id)

# --- Example Usage / Test Case ---
if __name__ == "__main__":
    import time

    def slow_processor(data):
        time.sleep(0.1)  # Simulate heavy work
        return data * 2

    queue = JobQueue()

    # Add jobs
    queue.add_job(1, 10)
    queue.add_job(2, 20)

    # Define a worker function for threads
    def worker(job_id):
        success = queue.process_job(job_id, slow_processor)
        print(f"Job {job_id} processed: {success}")

    # Create multiple threads trying to process the same jobs
    # In the buggy version, this would trigger KeyErrors
    threads = []
    for i in [1, 1, 2, 2]:  # Attempting to process job 1 and 2 twice each
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print(f"Results: {queue.get_result(1)}, {queue.get_result(2)}")