import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # The lock protects access to the shared dictionaries
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        with self._lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # 1. Atomically "claim" the job. 
        # By using pop() inside the lock, we ensure that only one thread 
        # can ever retrieve the data for a specific job_id.
        with self._lock:
            data = self.jobs.pop(job_id, None)

        # If data is None, the job wasn't in the dict or was already claimed
        if data is None:
            return False

        # 2. Execute the processor OUTSIDE the lock.
        # This is critical: if the processor takes a long time, we don't 
        # want to block other threads from adding or claiming other jobs.
        result = processor(data)

        # 3. Atomically store the result
        with self._lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self._lock:
            return self.results.get(job_id)

# --- Example Usage / Test ---
if __name__ == "__main__":
    import time

    queue = JobQueue()

    def slow_processor(data):
        time.sleep(0.5)  # Simulate heavy work
        return data * 2

    def worker(job_id):
        success = queue.process_job(job_id, slow_processor)
        print(f"Job {job_id} processing success: {success}")

    # Add a job
    queue.add_job(1, 10)

    # Create multiple threads trying to process the SAME job
    threads = []
    for i in range(5):
        t = threading.Thread(target=worker, args=(1,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print(f"Final Result: {queue.get_result(1)}")