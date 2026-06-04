import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to protect access to the shared dictionaries
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        with self._lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # 1. Atomically check if the job exists and remove it from the queue
        # This prevents multiple threads from processing the same job_id.
        with self._lock:
            if job_id not in self.jobs:
                return False
            
            # .pop() removes the item and returns it in one atomic-like step 
            # while we hold the lock.
            data = self.jobs.pop(job_id)

        # 2. Run the processor OUTSIDE the lock.
        # This is crucial: if the processor takes a long time, we don't 
        # want to block other threads from adding or retrieving jobs.
        result = processor(data)

        # 3. Atomically store the result
        with self._lock:
            self.results[job_id] = result
        
        return True

    def get_result(self, job_id):
        with self._lock:
            return self.results.get(job_id)

# --- Example Usage / Test ---
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

    # Create threads to process the same job simultaneously to test the fix
    t1 = threading.Thread(target=worker, args=(1,))
    t2 = threading.Thread(target=worker, args=(1,)) # Duplicate job call
    t3 = threading.Thread(target=worker, args=(2,))

    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()

    print(f"Result for Job 1: {queue.get_result(1)}")
    print(f"Result for Job 2: {queue.get_result(2)}")