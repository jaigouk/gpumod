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
        # 1. Attempt to "claim" the job atomically
        with self.lock:
            if job_id not in self.jobs:
                return False

            # We pop the job immediately. This ensures that even if another 
            # thread enters this method, the job_id is already gone from 
            # self.jobs, preventing double-processing.
            data = self.jobs.pop(job_id)

        # 2. Perform the actual processing OUTSIDE the lock.
        # This allows other threads to add or process other jobs 
        # while this thread is busy with the computation.
        result = processor(data)

        # 3. Save the result atomically
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)

# --- Example Usage / Test Case ---
if __name__ == "__main__":
    import time

    queue = JobQueue()

    def slow_processor(data):
        time.sleep(0.1)  # Simulate heavy work
        return data * 2

    def worker(job_id):
        success = queue.process_job(job_id, slow_processor)
        print(f"Job {job_id} processed: {success}")

    # Add a job
    queue.add_job(1, 10)

    # Start multiple threads trying to process the same job simultaneously
    threads = []
    for _ in range(5):
        t = threading.Thread(target=worker, args=(1,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print(f"Final Result: {queue.get_result(1)}")