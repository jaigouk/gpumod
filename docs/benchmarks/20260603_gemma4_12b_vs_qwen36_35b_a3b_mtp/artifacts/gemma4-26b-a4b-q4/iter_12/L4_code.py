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
        # 1. Acquire lock to safely extract the job data
        with self.lock:
            if job_id not in self.jobs:
                return False

            # Pop the job immediately. This ensures no other thread 
            # can pick up this same job_id while we are processing it.
            data = self.jobs.pop(job_id)

        # 2. Perform the heavy lifting OUTSIDE the lock.
        # This allows other threads to add/process other jobs concurrently.
        result = processor(data)

        # 3. Re-acquire lock to safely save the result
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)

# --- Testing the implementation ---
if __name__ == "__main__":
    import time

    def slow_processor(data):
        time.sleep(1)  # Simulate heavy work
        return data * 2

    queue = JobQueue()
    queue.add_job(1, 10)
    queue.add_job(2, 20)

    def worker(job_id):
        print(f"Thread starting job {job_id}")
        if queue.process_job(job_id, slow_processor):
            print(f"Thread finished job {job_id}")
        else:
            print(f"Thread failed job {job_id} (already taken or non-existent)")

    # Start multiple threads
    # We attempt to process job 1 twice to test the race condition fix
    t1 = threading.Thread(target=worker, args=(1,))
    t2 = threading.Thread(target=worker, args=(1,)) 
    t3 = threading.Thread(target=worker, args=(2,))

    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()

    print(f"Results: {queue.results}")