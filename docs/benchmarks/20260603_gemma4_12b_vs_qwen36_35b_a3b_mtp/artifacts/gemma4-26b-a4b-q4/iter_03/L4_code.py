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
        # 1. Atomically check if the job exists and "claim" it by popping it
        # This prevents two threads from picking up the same job_id.
        with self._lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id)

        # 2. Perform the heavy lifting OUTSIDE the lock.
        # This allows other threads to add/process other jobs simultaneously.
        try:
            result = processor(data)
            processing_failed = False
        except Exception as e:
            print(f"Error processing job {job_id}: {e}")
            processing_failed = True
            result = e

        # 3. Atomically store the result
        with self._lock:
            if not processing_failed:
                self.results[job_id] = result
            return True

    def get_result(self, job_id):
        with self._lock:
            return self.results.get(job_id)

# --- Example Usage / Test Case ---
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
        success = queue.process_job(job_id, slow_processor)
        print(f"Thread finished job {job_id}: {'Success' if success else 'Failed/Already taken'}")

    # Try to process the same job from two different threads simultaneously
    t1 = threading.Thread(target=worker, args=(1,))
    t2 = threading.Thread(target=worker, args=(1,))
    t3 = threading.Thread(target=worker, args=(2,))

    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()

    print(f"Result for Job 1: {queue.get_result(1)}")
    print(f"Result for Job 2: {queue.get_result(2)}")