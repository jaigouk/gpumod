import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # A lock to protect access to the internal dictionaries
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        with self._lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # 1. Atomically check and "claim" the job
        with self._lock:
            if job_id not in self.jobs:
                return False
            
            # pop() removes the job from the dict immediately.
            # This ensures no other thread can claim this job_id.
            data = self.jobs.pop(job_id)

        # 2. Run the processor OUTSIDE the lock.
        # If we held the lock here, the entire queue would become 
        # single-threaded, defeating the purpose of concurrency.
        try:
            result = processor(data)
            job_success = True
        except Exception as e:
            result = e
            job_success = False

        # 3. Atomically store the result
        with self._lock:
            self.results[job_id] = result
            
        return job_success

    def get_result(self, job_id):
        with self._lock:
            return self.results.get(job_id)

# --- Example Usage/Test ---
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
            print(f"Thread failed to claim job {job_id} (already taken)")

    # Create threads trying to process the same jobs
    threads = []
    for _ in range(4):
        threads.append(threading.Thread(target=worker, args=(1,)))
        threads.append(threading.Thread(target=worker, args=(2,)))

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    print(f"Results: {queue.results}")