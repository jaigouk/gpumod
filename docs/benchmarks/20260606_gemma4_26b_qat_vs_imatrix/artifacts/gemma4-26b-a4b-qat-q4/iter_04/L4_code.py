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
        # 1. ATOMIC CLAIM: We use the lock to check existence and 
        # remove the job from the queue in one step.
        with self.lock:
            if job_id not in self.jobs:
                return False
            # .pop() removes the item and returns it, ensuring no other 
            # thread can "claim" this specific job_id.
            data = self.jobs.pop(job_id)

        # 2. PROCESSING: We perform the heavy lifting OUTSIDE the lock.
        # This allows other threads to add/process other jobs concurrently.
        try:
            result = processor(data)
            job_success = True
        except Exception as e:
            result = e
            job_success = False

        # 3. STORE RESULT: Re-acquire the lock to update the results dictionary.
        with self.lock:
            self.results[job_id] = result

        return job_success

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)

# --- Demonstration of the fix ---
if __name__ == "__main__":
    import time

    queue = JobQueue()

    def slow_processor(data):
        time.sleep(1)  # Simulate heavy work
        return data * 2

    # Adding a job
    queue.add_job("task1", 10)

    # Simulate two threads trying to process the same job simultaneously
    def worker():
        success = queue.process_job("task1", slow_processor)
        print(f"Worker success: {success}")

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    print(f"Final Result: {queue.get_result('task1')}")