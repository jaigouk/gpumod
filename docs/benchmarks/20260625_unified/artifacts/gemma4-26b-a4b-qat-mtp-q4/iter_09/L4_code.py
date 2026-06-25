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
        # 1. ATOMIC CLAIM: We use the lock to check and remove the job 
        # from the queue in one single step. This prevents two threads 
        # from grabbing the same job.
        with self._lock:
            if job_id not in self.jobs:
                return False
            # .pop() removes the item and returns it, effectively "claiming" it
            data = self.jobs.pop(job_id)

        # 2. CONCURRENT EXECUTION: We call the processor OUTSIDE the lock.
        # This allows other threads to call add_job or process_job 
        # for OTHER jobs while this one is calculating.
        result = processor(data)

        # 3. ATOMIC WRITE: Lock again to safely save the result.
        with self._lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self._lock:
            return self.results.get(job_id)

# --- Example Usage / Testing ---
if __name__ == "__main__":
    import time

    def slow_processor(data):
        time.sleep(1)  # Simulate heavy work
        return data * 2

    queue = JobQueue()

    # Add jobs
    for i in range(5):
        queue.add_job(i, i)

    def worker(job_id):
        success = queue.process_job(job_id, slow_processor)
        print(f"Job {job_id} processed: {success}")

    # Launch multiple threads trying to process the same and different jobs
    threads = []
    for i in range(5):
        # We launch each job once
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

        # We also launch a "duplicate" thread to try and trigger the race condition
        # In the old code, this would likely cause a KeyError
        t_dup = threading.Thread(target=worker, args=(i,))
        threads.append(t_dup)
        t_dup.start()

    for t in threads:
        t.join()

    print("Final Results:", queue.results)