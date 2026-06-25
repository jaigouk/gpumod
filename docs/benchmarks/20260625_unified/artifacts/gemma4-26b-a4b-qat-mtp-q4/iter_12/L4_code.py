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
        # STEP 1: Atomically check if the job exists and "claim" it
        # We use pop() inside the lock so that no other thread can 
        # pick up this same job_id.
        with self.lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id)

        # STEP 2: Perform the heavy lifting OUTSIDE the lock.
        # This allows other threads to call add_job, get_result, 
        # or process other jobs simultaneously.
        try:
            result = processor(data)
        except Exception as e:
            # Handle potential errors in processing if necessary
            print(f"Error processing job {job_id}: {e}")
            return False

        # STEP 3: Atomically store the result
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)

# --- Example Usage / Test ---
if __name__ == "__main__":
    import time

    def slow_processor(data):
        time.sleep(1)  # Simulate a long-running task
        return data * 2

    queue = JobQueue()
    queue.add_job(1, 10)
    queue.add_job(2, 20)

    def worker(qid):
        success = queue.process_job(qid, slow_processor)
        print(f"Job {qid} processing success: {success}")

    # Start multiple threads trying to process the SAME job (to test the fix)
    # and DIFFERENT jobs (to test concurrency)
    threads = [
        threading.Thread(target=worker, args=(1,)),
        threading.Thread(target=worker, args=(1,)), # Duplicate call
        threading.Thread(target=worker, args=(2,)),
    ]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    print(f"Final Results: {queue.get_result(1)}, {queue.get_result(2)}")