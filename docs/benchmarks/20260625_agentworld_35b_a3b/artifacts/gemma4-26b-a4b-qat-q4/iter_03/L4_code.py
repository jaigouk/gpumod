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
        # 1. Atomically check existence and remove the job from the queue.
        # This prevents multiple threads from picking up the same job.
        with self._lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id)

        # 2. Execute the processor OUTSIDE the lock.
        # If we held the lock here, only one job could run at a time, 
        # defeating the purpose of multi-threading.
        try:
            result = processor(data)
            success = True
        except Exception as e:
            result = e
            success = False

        # 3. Atomically store the result.
        with self._lock:
            self.results[job_id] = result

        return success

    def get_result(self, job_id):
        with self._lock:
            return self.results.get(job_id)

# --- Testing the fix ---
if __name__ == "__main__":
    import time

    def slow_processor(data):
        time.sleep(1)  # Simulate heavy work
        return data * 2

    queue = JobQueue()
    queue.add_job(1, 10)
    queue.add_job(1, 10)  # Duplicate ID to test race condition
    queue.add_job(2, 20)

    threads = []
    # We attempt to process job 1 multiple times via different threads
    for i in range(5):
        t = threading.Thread(target=queue.process_job, args=(1, slow_processor))
        threads.append(t)
        t.start()

    # Process job 2
    t2 = threading.Thread(target=queue.process_job, args=(2, slow_processor))
    threads.append(t2)
    t2.start()

    for t in threads:
        t.join()

    print(f"Result 1: {queue.get_result(1)}") # Should be 20
    print(f"Result 2: {queue.get_result(2)}") # Should be 40