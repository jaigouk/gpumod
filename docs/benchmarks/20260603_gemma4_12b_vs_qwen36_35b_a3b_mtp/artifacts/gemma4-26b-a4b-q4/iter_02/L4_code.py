import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        with self._lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # 1. Atomically "claim" the job
        # We use pop() inside the lock so that only one thread 
        # can ever retrieve the data for a specific job_id.
        with self._lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id)

        # 2. Process the job 
        # We do this OUTSIDE the lock so other threads can 
        # add/process other jobs while this one is working.
        try:
            result = processor(data)
            job_success = True
        except Exception as e:
            result = e
            job_success = False

        # 3. Atomically save the result
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
        time.sleep(0.1)  # Simulate heavy work
        return data * 2

    queue = JobQueue()
    queue.add_job(1, 10)

    def worker():
        # Multiple threads try to process the same job
        success = queue.process_job(1, slow_processor)
        print(f"Thread {threading.current_thread().name} success: {success}")

    threads = [threading.Thread(target=worker, name=f"T{i}") for i in range(5)]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"Final Result for job 1: {queue.get_result(1)}")