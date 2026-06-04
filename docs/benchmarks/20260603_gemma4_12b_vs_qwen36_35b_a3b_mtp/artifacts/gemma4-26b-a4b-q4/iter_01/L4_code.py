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
        # 1. Extract the data and remove it from the jobs dict atomically.
        # This prevents multiple threads from picking up the same job.
        with self._lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id)

        # 2. Run the processor OUTSIDE the lock.
        # If we held the lock here, no other thread could add or process jobs
        # while this specific job is running.
        result = processor(data)

        # 3. Store the result atomically.
        with self._lock:
            self.results[job_id] = result
            
        return True

    def get_result(self, job_id):
        with self._lock:
            return self.results.get(job_id)

# --- Demonstration of usage ---
if __name__ == "__main__":
    import time

    def slow_processor(data):
        time.sleep(1)  # Simulate heavy work
        return data * 2

    queue = JobQueue()
    queue.add_job(1, 10)
    queue.add_job(2, 20)

    def worker(job_id):
        print(f"Starting job {job_id}")
        if queue.process_job(job_id, slow_processor):
            print(f"Finished job {job_id}")
        else:
            print(f"Job {job_id} failed or already taken")

    # Create threads to process jobs
    t1 = threading.Thread(target=worker, args=(1,))
    t2 = threading.Thread(target=worker, args=(1,)) # Attempting to process same job
    t3 = threading.Thread(target=worker, args=(2,))

    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()

    print(f"Result for job 1: {queue.get_result(1)}")
    print(f"Result for job 2: {queue.get_result(2)}")