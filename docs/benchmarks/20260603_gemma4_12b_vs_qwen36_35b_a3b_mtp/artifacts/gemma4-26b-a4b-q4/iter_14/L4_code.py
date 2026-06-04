import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # A lock to protect access to the shared dictionaries
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        with self._lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # 1. Atomic Check-and-Remove
        # We use the lock to safely check if the job exists and remove it.
        # Removing it immediately (via pop) ensures no other thread can 
        # pick up this same job_id.
        with self._lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id)

        # 2. Processing (Performed OUTSIDE the lock)
        # We release the lock while the processor is running. 
        # This allows other threads to add jobs or process other jobs 
        # in parallel.
        try:
            result = processor(data)
            processing_successful = True
        except Exception as e:
            result = e
            processing_successful = False

        # 3. Atomic Write Result
        # We re-acquire the lock to safely write the result to the dictionary.
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
        print(f"Thread attempting job {job_id}")
        if queue.process_job(job_id, slow_processor):
            print(f"Thread finished job {job_id}")
        else:
            print(f"Thread failed to claim job {job_id}")

    # Create two threads attempting to process the SAME job
    # One should succeed, the other should fail immediately.
    t1 = threading.Thread(target=worker, args=(1,))
    t2 = threading.Thread(target=worker, args=(1,))

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    print(f"Result for job 1: {queue.get_result(1)}")