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
        # Step 1: Atomic "Claim"
        # We use the lock to check if the job exists and remove it 
        # from the queue so no other thread can pick it up.
        with self.lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id)

        # Step 2: Processing
        # We perform the processing OUTSIDE the lock. 
        # This allows other threads to call add_job or process_job 
        # while this thread is working.
        result = processor(data)

        # Step 3: Atomic Result Storage
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)

# --- Testing the fix ---
if __name__ == "__main__":
    import time

    def slow_processor(data):
        time.sleep(0.1)  # Simulate heavy work
        return data * 2

    queue = JobQueue()
    queue.add_job(1, 10)
    queue.add_job(1, 10) # Attempting to add same ID again

    threads = []
    for _ in range(5):
        t = threading.Thread(target=queue.process_job, args=(1, slow_processor))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print(f"Result: {queue.get_result(1)}")
    print(f"Jobs remaining: {len(queue.jobs)}")