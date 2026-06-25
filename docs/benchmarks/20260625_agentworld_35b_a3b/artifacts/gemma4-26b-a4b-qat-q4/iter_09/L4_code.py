import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # A lock to protect access to the shared dictionaries
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # ATOMIC STEP: Try to claim the job by popping it from the dictionary.
        # This prevents multiple threads from processing the same job_id.
        with self.lock:
            data = self.jobs.pop(job_id, None)

        if data is None:
            return False

        # Perform the heavy lifting OUTSIDE the lock so we don't block 
        # other threads from adding or claiming different jobs.
        result = processor(data)

        # ATOMIC STEP: Save the result
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)

# --- Testing the fix ---
if __name__ == "__main__":
    import time

    def heavy_task(n):
        time.sleep(0.1)  # Simulate work
        return n * n

    queue = JobQueue()

    def worker(job_id):
        # Attempt to process the job
        success = queue.process_job(job_id, heavy_task)
        if success:
            print(f"Job {job_id} processed successfully.")
        else:
            print(f"Job {job_id} was already taken or doesn't exist.")

    # Add jobs
    for i in range(5):
        queue.add_job(i, i)

    # Create multiple threads trying to process the SAME job_id to test race condition
    threads = []
    for _ in range(10):
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

    for t in threads:
        t.join()

    print("Results:", queue.results)