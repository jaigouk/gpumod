import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Create a lock to protect access to the shared dictionaries
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        with self._lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # 1. Atomically check if the job exists and "claim" it by removing it
        # from the jobs dictionary. This prevents other threads from processing 
        # the same job.
        with self._lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id)

        # 2. Run the processor OUTSIDE the lock. 
        # If we held the lock here, no other thread could add or get results
        # while the processor is running, defeating the purpose of concurrency.
        result = processor(data)

        # 3. Atomically store the result
        with self._lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self._lock:
            return self.results.get(job_id)

# Example Usage & Verification
if __name__ == "__main__":
    import time

    def heavy_task(n):
        time.sleep(1)  # Simulate work
        return n * n

    queue = JobQueue()

    # Adding a job
    queue.add_job(1, 10)

    # Simulating two threads trying to process the SAME job simultaneously
    def worker(job_id):
        success = queue.process_job(job_id, heavy_task)
        print(f"Thread {threading.current_thread().name} success: {success}")

    t1 = threading.Thread(target=worker, args=(1,), name="A")
    t2 = threading.Thread(target=worker, args=(1,), name="B")

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    print(f"Final Result: {queue.get_result(1)}")