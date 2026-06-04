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
        # 1. Acquire lock to safely extract the job
        with self._lock:
            # pop() removes the key and returns the value in one atomic step
            # If job_id isn't there, it returns None
            data = self.jobs.pop(job_id, None)

        # If data is None, the job wasn't in the queue (or was already taken)
        if data is None:
            return False

        # 2. Process the job OUTSIDE the lock.
        # This allows other threads to call add_job or process_job 
        # while this specific thread is doing heavy lifting.
        result = processor(data)

        # 3. Acquire lock again to safely save the result
        with self._lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self._lock:
            return self.results.get(job_id)

# --- Verification Logic ---
if __name__ == "__main__":
    import time

    def slow_processor(data):
        time.sleep(0.1)  # Simulate work
        return data * 2

    queue = JobQueue()

    # Add a job
    queue.add_job(1, 10)

    # Simulate two threads trying to process the SAME job simultaneously
    def worker():
        success = queue.process_job(1, slow_processor)
        print(f"Thread {threading.current_thread().name} success: {success}")

    t1 = threading.Thread(target=worker, name="A")
    t2 = threading.Thread(target=worker, name="B")

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    print(f"Final Result for Job 1: {queue.get_result(1)}")