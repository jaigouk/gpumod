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
        # 1. ATOMIC CHECK AND REMOVE
        # We use the lock to ensure only ONE thread can "claim" this job_id.
        with self.lock:
            if job_id not in self.jobs:
                return False

            # We pop the job immediately so no other thread can pick it up
            data = self.jobs.pop(job_id)

        # 2. HEAVY PROCESSING
        # We perform the actual work OUTSIDE the lock.
        # This allows other threads to add jobs or process other jobs simultaneously.
        try:
            result = processor(data)
            processing_successful = True
        except Exception as e:
            result = e
            processing_successful = False

        # 3. ATOMIC RESULT STORAGE
        # We re-acquire the lock to safely write the result.
        with self.lock:
            self.results[job_id] = result

        return processing_successful

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)

# --- Testing the fix ---
if __name__ == "__main__":
    import time

    def slow_processor(data):
        time.sleep(1)  # Simulate heavy work
        return data * 2

    queue = JobQueue()
    queue.add_job(1, 10)
    queue.add_job(2, 20)

    # Start multiple threads trying to process the same job ID
    # One should succeed, others should return False.
    def worker(job_id):
        success = queue.process_job(job_id, slow_processor)
        print(f"Job {job_id} processed by thread {threading.current_thread().name}: {success}")

    threads = []
    # Thread A and B both try to grab Job 1
    t1 = threading.Thread(target=worker, args=(1,), name="Worker-A")
    t2 = threading.Thread(target=worker, args=(1,), name="Worker-B")
    t3 = threading.Thread(target=worker, args=(2,), name="Worker-C")

    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()

    print(f"Final Results: {queue.results}")