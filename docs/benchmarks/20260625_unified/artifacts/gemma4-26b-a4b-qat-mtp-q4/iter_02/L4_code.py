import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # A lock to protect access to the dictionaries
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        with self._lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # 1. Critical Section: Check existence and remove job from the queue
        # We use .pop() inside the lock to ensure that only one thread 
        # can ever "claim" this job for processing.
        with self._lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id)

        # 2. Non-Critical Section: Perform the actual work
        # We perform this OUTSIDE the lock so that other threads can 
        # call add_job() or process_job() while this thread is busy.
        try:
            result = processor(data)
            processing_success = True
        except Exception:
            # Handle potential errors in the processor if necessary
            processing_success = False
            result = None

        # 3. Critical Section: Store the result
        with self._lock:
            self.results[job_id] = result

        return processing_success

    def get_result(self, job_id):
        with self._lock:
            return self.results.get(job_id)

# --- Example Usage / Testing ---
if __name__ == "__main__":
    import time

    def slow_processor(data):
        time.sleep(1)  # Simulate heavy work
        return data * 2

    queue = JobQueue()

    # Add a job
    queue.add_job(1, 10)
    queue.add_job(2, 20)

    # Define a worker function for threads
    def worker(job_id):
        success = queue.process_job(job_id, slow_processor)
        print(f"Job {job_id} processed: {success}")

    # Start multiple threads trying to process the SAME job ID
    # In the old version, this would cause a KeyError.
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