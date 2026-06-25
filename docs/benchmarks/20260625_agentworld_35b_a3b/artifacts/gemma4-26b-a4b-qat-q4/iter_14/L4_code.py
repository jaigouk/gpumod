import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Create a lock to synchronize access to the dictionaries
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        with self._lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # 1. ATOMIC CLAIM: Try to remove the job from the queue immediately.
        # This ensures only ONE thread can ever "own" this job_id for processing.
        with self._lock:
            data = self.jobs.pop(job_id, None)

        # If data is None, the job didn't exist or was already claimed/processed
        if data is None:
            return False

        # 2. EXECUTION: Run the processor OUTSIDE the lock.
        # It is crucial to release the lock during the heavy processing so 
        # other threads can add/process other jobs simultaneously.
        result = processor(data)

        # 3. SAFE WRITE: Store the result under the lock.
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
    queue.add_job(1, 10)

    # Simulate multiple threads trying to process the same job ID
    def worker(q, jid, proc):
        success = q.process_job(jid, proc)
        print(f"Worker success: {success}")

    threads = []
    for _ in range(5):
        t = threading.Thread(target=worker, args=(queue, 1, slow_processor))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print(f"Final Result: {queue.get_result(1)}")