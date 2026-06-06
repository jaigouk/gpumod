import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # A lock to ensure atomic access to the shared dictionaries
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # 1. Atomically check existence and "claim" the job
        # We use .pop() inside the lock to remove the job immediately.
        # This ensures no other thread can pick up this specific job_id.
        with self.lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id)

        # 2. Run the heavy processing work OUTSIDE the lock.
        # This allows other threads to add_job or process other jobs concurrently.
        result = processor(data)

        # 3. Atomically store the result
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)

# Example Usage / Test Case
if __name__ == "__main__":
    import time

    def slow_processor(data):
        time.sleep(1)  # Simulate heavy work
        return data * 2

    queue = JobQueue()
    queue.add_job(1, 10)
    queue.add_job(2, 20)

    def worker(job_id):
        success = queue.process_job(job_id, slow_processor)
        print(f"Job {job_id} processed: {success}")

    # Create multiple threads attempting to process the SAME job ID
    # In the original code, this would cause a KeyError or double processing
    threads = [
        threading.Thread(target=worker, args=(1,)),
        threading.Thread(target=worker, args=(1,)),
        threading.Thread(target=worker, args=(2,)),
    ]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    print(f"Results: {queue.get_result(1)}") # Should be 20
    print(f"Results: {queue.get_result(2)}") # Should be 40