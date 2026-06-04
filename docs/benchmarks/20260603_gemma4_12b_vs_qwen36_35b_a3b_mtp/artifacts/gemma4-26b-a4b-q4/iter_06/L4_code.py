import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # A single lock to protect access to both shared dictionaries
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        with self._lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # 1. Securely retrieve and remove the job from the queue
        # We "pop" the job while holding the lock so that no other 
        # thread can pick up the same job_id.
        with self._lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id)

        # 2. Run the processor OUTSIDE the lock.
        # This is crucial! If we held the lock here, only one thread 
        # could process a job at a time, defeating the purpose of concurrency.
        result = processor(data)

        # 3. Securely store the result
        with self._lock:
            self.results[job_id] = result
        
        return True

    def get_result(self, job_id):
        with self._lock:
            return self.results.get(job_id)

# --- Implementation Test ---
if __name__ == "__main__":
    import time

    def slow_processor(data):
        time.sleep(1)  # Simulate heavy work
        return data * 2

    queue = JobQueue()
    
    # Add jobs
    queue.add_job(1, 10)
    queue.add_job(2, 20)

    def worker(job_id):
        print(f"Thread starting job {job_id}")
        success = queue.process_job(job_id, slow_processor)
        print(f"Thread finished job {job_id}: {success}")

    # Start two threads
    t1 = threading.Thread(target=worker, args=(1,))
    t2 = threading.Thread(target=worker, args=(2,))
    
    # Attempt to start a duplicate thread for job 1 to test race condition
    t3 = threading.Thread(target=worker, args=(1,))

    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()

    print(f"Result for job 1: {queue.get_result(1)}")
    print(f"Result for job 2: {queue.get_result(2)}")