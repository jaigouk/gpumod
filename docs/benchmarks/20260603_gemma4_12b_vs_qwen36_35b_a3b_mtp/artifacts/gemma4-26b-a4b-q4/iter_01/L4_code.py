import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # A lock to protect access to the shared dictionaries
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # 1. ATOMIC CHECK AND CLAIM
        # We use the lock to safely check if the job exists and 
        # "claim" it by removing it from the jobs dict immediately.
        with self.lock:
            if job_id not in self.jobs:
                return False

            # .pop() removes the item and returns it in one atomic step 
            # relative to our lock, ensuring no other thread can claim it.
            data = self.jobs.pop(job_id)

        # 2. EXECUTION
        # We perform the actual processing OUTSIDE the lock.
        # This allows other threads to add jobs or process other jobs 
        # while this thread is busy with heavy computation/IO.
        try:
            result = processor(data)
            processing_success = True
        except Exception as e:
            result = e
            processing_success = False

        # 3. ATOMIC RESULT STORAGE
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)

# --- Example Usage/Test ---
if __name__ == "__main__":
    import time

    def slow_processor(data):
        time.sleep(0.1)  # Simulate work
        return data * 2

    queue = JobQueue()
    queue.add_job(1, 10)
    queue.add_job(2, 20)

    def worker(job_id):
        success = queue.process_job(job_id, slow_processor)
        print(f"Job {job_id} processed: {success}")

    # Create multiple threads trying to process the same job
    # Only one should succeed, and no KeyError should be raised.
    threads = []
    for _ in range(5):
        threads.append(threading.Thread(target=worker, args=(1,)))
        threads.append(threading.Thread(target=worker, args=(2,)))

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"Result for Job 1: {queue.get_result(1)}")
    print(f"Result for Job 2: {queue.get_result(2)}")