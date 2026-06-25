import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # A lock to protect access to the jobs and results dictionaries
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        with self._lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # 1. ATOMIC CLAIM: Check if job exists and remove it from 'jobs' 
        # in one atomic step to ensure only one thread can claim it.
        with self._lock:
            if job_id not in self.jobs:
                return False
            # pop() removes the item and returns it, effectively "claiming" the job
            data = self.jobs.pop(job_id)

        # 2. PROCESSING: Perform the heavy lifting OUTSIDE the lock.
        # Holding the lock during the processor call would stop all other threads.
        try:
            result = processor(data)
            success = True
        except Exception:
            # Depending on requirements, you might handle errors differently.
            # For this fix, we assume successful processing.
            success = False
            result = None

        # 3. ATOMIC RESULT: Save the result to the dictionary.
        with self._lock:
            self.results[job_id] = result

        return success

    def get_result(self, job_id):
        with self._lock:
            return self.results.get(job_id)

# Example Usage/Test
if __name__ == "__main__":
    import time

    def slow_processor(data):
        time.sleep(0.1)  # Simulate work
        return data * 2

    queue = JobQueue()
    queue.add_job(1, 10)
    queue.add_job(2, 20)

    def worker(job_id):
        if queue.process_job(job_id, slow_processor):
            print(f"Job {job_id} completed.")
        else:
            print(f"Job {job_id} already processed or not found.")

    # Start multiple threads trying to process the same job
    threads = []
    for _ in range(5):
        threads.append(threading.Thread(target=worker, args=(1,)))
        threads.append(threading.Thread(target=worker, args=(2,)))

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"Results: {queue.get_result(1)}, {queue.get_result(2)}")