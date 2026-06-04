import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # A lock to protect access to the jobs and results dictionaries
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        with self._lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # 1. ATOMIC CLAIM: 
        # We use the lock to check existence and remove the job in one step.
        # This prevents two threads from processing the same job.
        with self._lock:
            if job_id not in self.jobs:
                return False
            # .pop() removes the item and returns it, ensuring no other 
            # thread can see this job_id in self.jobs anymore.
            data = self.jobs.pop(job_id)

        # 2. CONCURRENT PROCESSING:
        # We perform the heavy lifting OUTSIDE the lock. 
        # This allows other threads to add/process other jobs simultaneously.
        result = processor(data)

        # 3. ATOMIC STORE:
        # Re-acquire the lock to safely write the result.
        with self._lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self._lock:
            return self.results.get(job_id)

# --- Demonstration of usage ---
if __name__ == "__main__":
    import time

    def heavy_processor(data):
        time.sleep(1)  # Simulate expensive work
        return data * 2

    queue = JobQueue()
    queue.add_job(1, 10)
    queue.add_job(2, 20)

    def worker(jid):
        print(f"Thread {threading.current_thread().name} starting job {jid}")
        if queue.process_job(jid, heavy_processor):
            print(f"Thread {threading.current_thread().name} finished job {jid}")
        else:
            print(f"Thread {threading.current_thread().name} failed job {jid} (already taken)")

    # Create threads to attempt to process the same job and different jobs
    t1 = threading.Thread(target=worker, args=(1,), name="Worker-1")
    t2 = threading.Thread(target=worker, args=(1,), name="Worker-2") # Attempt duplicate
    t3 = threading.Thread(target=worker, args=(2,), name="Worker-3")

    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()

    print(f"Result for job 1: {queue.get_result(1)}")
    print(f"Result for job 2: {queue.get_result(2)}")