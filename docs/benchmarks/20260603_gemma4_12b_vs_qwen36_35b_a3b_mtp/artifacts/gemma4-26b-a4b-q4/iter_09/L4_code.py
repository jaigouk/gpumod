import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to synchronize access to shared state
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # 1. ATOMIC CLAIM: Check if job exists and remove it immediately
        # We do this inside the lock so no two threads can "claim" the same job.
        with self.lock:
            if job_id not in self.jobs:
                return False
            # pop() removes the item and returns it in one atomic-like step
            data = self.jobs.pop(job_id)

        # 2. PROCESSING: Perform the actual work OUTSIDE the lock.
        # This allows other threads to add/process other jobs while this one runs.
        try:
            result = processor(data)
            work_successful = True
        except Exception as e:
            result = e
            work_successful = False

        # 3. ATOMIC WRITE: Save the result
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
    queue.add_job(2, 20)

    def worker(job_id):
        success = queue.process_job(job_id, slow_processor)
        print(f"Thread for job {job_id} success: {success}")

    # Start two threads trying to process the SAME job ID to test the race condition
    t1 = threading.Thread(target=worker, args=(1,))
    t2 = threading.Thread(target=worker, args=(1,))

    # Start another thread for a DIFFERENT job ID
    t3 = threading.Thread(target=worker, args=(2,))

    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()

    print(f"Result for Job 1: {queue.get_result(1)}")
    print(f"Result for Job 2: {queue.get_result(2)}")