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
        # 1. Check and "claim" the job atomically
        with self._lock:
            if job_id not in self.jobs:
                return False
            # .pop() removes the job from the queue immediately so 
            # no other thread can pick it up.
            data = self.jobs.pop(job_id)

        # 2. Perform the heavy lifting OUTSIDE the lock.
        # This allows other threads to process other jobs simultaneously.
        result = processor(data)

        # 3. Store the result safely
        with self._lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self._lock:
            return self.results.get(job_id)

# --- Demonstration of usage ---
if __name__ == "__main__":
    import time

    def heavy_task(n):
        time.sleep(1)  # Simulate work
        return n * 2

    queue = JobQueue()
    queue.add_job(1, 10)
    queue.add_job(2, 20)

    # Start threads to process jobs
    t1 = threading.Thread(target=queue.process_job, args=(1, heavy_task))
    t2 = threading.Thread(target=queue.process_job, args=(2, heavy_task))

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    print(f"Result 1: {queue.get_result(1)}")
    print(f"Result 2: {queue.get_result(2)}")