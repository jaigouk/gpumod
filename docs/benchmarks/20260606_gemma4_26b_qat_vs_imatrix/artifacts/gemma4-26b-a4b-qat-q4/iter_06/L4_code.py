import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Lock for protecting the jobs dictionary
        self.jobs_lock = threading.Lock()
        # Lock for protecting the results dictionary
        self.results_lock = threading.Lock()

    def add_job(self, job_id, data):
        with self.jobs_lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # 1. ATOMIC CLAIM: Use pop() inside a lock to ensure only 
        # ONE thread can ever "win" the right to process this job_id.
        with self.jobs_lock:
            data = self.jobs.pop(job_id, None)

        if data is None:
            return False

        # 2. PROCESSING: We perform the heavy lifting OUTSIDE the lock.
        # This allows other threads to add/process other jobs simultaneously.
        try:
            result = processor(data)
        except Exception as e:
            # Handle or log the error; here we just ensure the job doesn't 
            # hang the system if the processor fails.
            return False

        # 3. ATOMIC WRITE: Securely write the result to the results dict.
        with self.results_lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self.results_lock:
            return self.results.get(job_id)

# --- Example Usage/Test ---
if __name__ == "__main__":
    import time
    import random

    def slow_processor(data):
        time.sleep(random.uniform(0.1, 0.5))  # Simulate work
        return data * 2

    queue = JobQueue()

    def worker(job_id):
        if queue.process_job(job_id, slow_processor):
            print(f"Job {job_id} processed successfully.")
        else:
            print(f"Job {job_id} was already taken or doesn't exist.")

    # Add jobs
    for i in range(5):
        queue.add_job(i, i + 10)

    # Simulate multiple threads trying to process the same jobs
    threads = []
    for _ in range(15): # 15 threads competing for 5 jobs
        j_id = random.randint(0, 4)
        t = threading.Thread(target=worker, args=(j_id,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print("Final Results:", queue.get_result(0)) # Should be 20