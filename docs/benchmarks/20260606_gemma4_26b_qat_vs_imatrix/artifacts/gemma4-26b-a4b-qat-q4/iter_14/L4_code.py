import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # A single lock is sufficient to protect the integrity of the dictionaries
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        with self._lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # Step 1: "Claim" the job.
        # We lock only long enough to check if the job exists and remove it.
        # This ensures that only ONE thread can ever "own" a specific job_id.
        with self._lock:
            if job_id not in self.jobs:
                return False

            # .pop() removes the item and returns it, ensuring atomicity 
            # within the lock so no other thread can grab it.
            data = self.jobs.pop(job_id)

        # Step 2: Execute the processor.
        # IMPORTANT: We do NOT hold the lock during the processing step.
        # If we did, only one job could run at a time across the whole app.
        result = processor(data)

        # Step 3: Store the result.
        # We lock again to ensure thread-safe writing to the results dictionary.
        with self._lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self._lock:
            return self.results.get(job_id)

# --- Testing the fix ---
if __name__ == "__main__":
    import time

    def slow_processor(data):
        time.sleep(1)  # Simulate heavy work
        return data * 2

    queue = JobQueue()
    queue.add_job(1, 10)
    queue.add_job(2, 20)

    # Simulate multiple threads trying to process the same job ID simultaneously
    def worker(job_id):
        success = queue.process_job(job_id, slow_processor)
        print(f"Job {job_id} processed by {threading.current_thread().name}: {success}")

    t1 = threading.Thread(target=worker, args=(1,), name="Thread-1")
    t2 = threading.Thread(target=worker, args=(1,), name="Thread-2") # Duplicate job ID
    t3 = threading.Thread(target=worker, args=(2,), name="Thread-3")

    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()

    print(f"Final Results: {queue.get_result(1)}, {queue.get_result(2)}")