import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # A lock to protect access to the internal dictionaries
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # 1. ATOMIC CLAIM:
        # We use the lock to check if the job exists and remove it 
        # from the queue in one atomic step. This prevents two threads
        # from picking up the same job.
        with self.lock:
            if job_id not in self.jobs:
                return False
            # pop() removes the item and returns it, effectively "claiming" it
            data = self.jobs.pop(job_id)

        # 2. CONCURRENT PROCESSING:
        # We perform the heavy lifting OUTSIDE the lock. 
        # This allows other threads to add/process OTHER jobs simultaneously.
        result = processor(data)

        # 3. ATOMIC RESULT STORAGE:
        # We re-acquire the lock only to write the result to the shared dictionary.
        with self.lock:
            self.results[job_id] = result
        
        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)

# --- Example Usage/Test ---
if __name__ == "__main__":
    import time

    def heavy_task(n):
        time.sleep(1)  # Simulate work
        return n * n

    queue = JobQueue()
    queue.add_job(1, 10)
    queue.add_job(2, 20)

    def worker(job_id):
        success = queue.process_job(job_id, heavy_task)
        print(f"Job {job_id} processed: {success}")

    # Start two threads
    t1 = threading.Thread(target=worker, args=(1,))
    t2 = threading.Thread(target=worker, args=(1,)) # Attempting to process the same job
    t3 = threading.Thread(target=worker, args=(2,))

    t1.start()
    t2.start() # This one should return False because t1 "claimed" it
    t3.start()

    t1.join()
    t2.join()
    t3.join()

    print(f"Result for job 1: {queue.get_result(1)}")
    print(f"Result for job 2: {queue.get_result(2)}")