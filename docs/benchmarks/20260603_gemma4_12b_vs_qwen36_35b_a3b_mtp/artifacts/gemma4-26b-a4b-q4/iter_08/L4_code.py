import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # A single lock to protect access to the shared dictionaries
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # 1. ATOMIC CLAIM: 
        # We use the lock to check existence and remove the job in one step.
        # This ensures only ONE thread can ever "own" a specific job_id.
        with self.lock:
            if job_id not in self.jobs:
                return False
            
            # .pop() removes the item and returns it, preventing other 
            # threads from seeing it in self.jobs
            data = self.jobs.pop(job_id)

        # 2. WORK:
        # We perform the actual processing OUTSIDE the lock.
        # This allows other threads to call add_job or process_job 
        # for different IDs while this thread is busy.
        result = processor(data)

        # 3. ATOMIC STORE:
        # We re-acquire the lock to safely write the result.
        with self.lock:
            self.results[job_id] = result
            
        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)

# --- Example Usage / Test ---
if __name__ == "__main__":
    import time

    def slow_processor(data):
        time.sleep(0.1)  # Simulate heavy work
        return data * 2

    queue = JobQueue()
    queue.add_job(1, 10)
    queue.add_job(2, 20)

    def worker(job_id):
        success = queue.process_job(job_id, slow_processor)
        print(f"Job {job_id} processed: {success}")

    # Create multiple threads attempting to process the SAME job
    # to test the race condition fix.
    threads = []
    for _ in range(3):
        threads.append(threading.Thread(target=worker, args=(1,)))
    
    # Also add a thread for a different job
    threads.append(threading.Thread(target=worker, args=(2,)))

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"Result for Job 1: {queue.get_result(1)}")
    print(f"Result for Job 2: {queue.get_result(2)}")