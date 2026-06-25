import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # A lock to protect access to shared dictionaries
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        with self._lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # 1. Atomically "claim" the job by popping it from the dictionary.
        # This prevents multiple threads from processing the same job_id.
        with self._lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id)

        # 2. Perform the heavy lifting OUTSIDE the lock.
        # If we kept the lock here, the entire queue would freeze until 
        # this processor finishes, defeating the purpose of threading.
        try:
            result = processor(data)
            success = True
        except Exception as e:
            result = e
            success = False

        # 3. Atomically store the result.
        with self._lock:
            self.results[job_id] = result

        return success

    def get_result(self, job_id):
        with self._lock:
            return self.results.get(job_id)

# --- Example Usage / Test ---
if __name__ == "__main__":
    import time

    def slow_processor(data):
        time.sleep(0.5)  # Simulate heavy work
        return data * 2

    queue = JobQueue()
    queue.add_job("job1", 10)
    queue.add_job("job2", 20)

    # Start two threads trying to process the SAME job_id
    # One should succeed, the other should return False
    def worker(jid):
        res = queue.process_job(jid, slow_processor)
        print(f"Thread processing {jid}: {'Success' if res else 'Failed (Already taken)'}")

    t1 = threading.Thread(target=worker, args=("job1",))
    t2 = threading.Thread(target=worker, args=("job1",)) # Duplicate job attempt

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    print(f"Result for job1: {queue.get_result('job1')}")