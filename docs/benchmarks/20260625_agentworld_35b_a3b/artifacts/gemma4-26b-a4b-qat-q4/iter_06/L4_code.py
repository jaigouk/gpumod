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
        # 1. Atomically check and "claim" the job
        # We use pop() inside the lock to ensure only one thread can 
        # ever take this specific job_id for processing.
        with self._lock:
            data = self.jobs.pop(job_id, None)

        if data is None:
            return False

        # 2. Process the job OUTSIDE the lock
        # This allows other threads to add jobs or process other jobs simultaneously.
        try:
            result = processor(data)
            job_success = True
        except Exception:
            # Handle potential errors in the processor if necessary
            result = None
            job_success = False

        # 3. Atomically store the result
        with self._lock:
            if job_success:
                self.results[job_id] = result

        return job_success

    def get_result(self, job_id):
        with self._lock:
            return self.results.get(job_id)

# Example usage:
if __name__ == "__main__":
    import time

    def slow_processor(data):
        time.sleep(1)  # Simulate heavy work
        return data * 2

    queue = JobQueue()
    queue.add_job(1, 10)
    queue.add_job(2, 20)

    # Simulate multiple threads trying to process jobs
    t1 = threading.Thread(target=queue.process_job, args=(1, slow_processor))
    t2 = threading.Thread(target=queue.process_job, args=(1, slow_processor)) # Duplicate job attempt
    t3 = threading.Thread(target=queue.process_job, args=(2, slow_processor))

    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()

    print(f"Result for Job 1: {queue.get_result(1)}")
    print(f"Result for Job 2: {queue.get_result(2)}")