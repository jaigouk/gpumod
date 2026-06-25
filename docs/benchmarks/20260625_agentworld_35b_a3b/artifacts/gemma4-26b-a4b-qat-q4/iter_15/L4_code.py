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
        # 1. Atomically check and claim the job
        # We use pop() inside the lock so that only one thread 
        # can successfully grab the data for a specific job_id.
        with self._lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id)

        # 2. Execute the processor OUTSIDE the lock.
        # This allows other threads to call add_job or process_job 
        # for other IDs while this thread is busy calculating.
        try:
            result = processor(data)
            success = True
        except Exception:
            # Handle potential errors in the processor if necessary
            result = None
            success = False

        # 3. Atomically store the result
        with self._lock:
            self.results[job_id] = result

        return success

    def get_result(self, job_id):
        with self._lock:
            return self.results.get(job_id)

# --- Example Usage / Test Case ---
if __name__ == "__main__":
    import time

    def heavy_computation(data):
        time.sleep(1)  # Simulate a long-running task
        return data * 2

    queue = JobQueue()
    queue.add_job(1, 10)
    queue.add_job(2, 20)

    def worker(job_id):
        if queue.process_job(job_id, heavy_computation):
            print(f"Job {job_id} processed successfully.")
        else:
            print(f"Job {job_id} was already taken or doesn't exist.")

    # Simulate multiple threads trying to process the SAME job simultaneously
    threads = []
    for _ in range(5):
        # Thread 0 and 1 fight over Job 1
        threads.append(threading.Thread(target=worker, args=(1,)))
        # Thread 2 and 3 fight over Job 2
        threads.append(threading.Thread(target=worker, args=(2,)))

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"Final Results: {queue.get_result(1)}, {queue.get_result(2)}")