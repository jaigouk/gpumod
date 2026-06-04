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
        # 1. Acquire lock to safely check and "claim" the job
        with self.lock:
            if job_id not in self.jobs:
                return False

            # .pop() removes the job immediately so no other thread 
            # can pick it up, even while we are processing it.
            data = self.jobs.pop(job_id)

        # 2. Run the processor OUTSIDE the lock.
        # This allows other threads to call add_job or process_job
        # for different IDs while this thread is busy computing.
        try:
            result = processor(data)
            success = True
        except Exception as e:
            result = e
            success = False

        # 3. Acquire lock again to safely write the result
        with self.lock:
            self.results[job_id] = result

        return success

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)

# --- Demonstration of the fix ---
if __name__ == "__main__":
    import time

    def slow_processor(data):
        time.sleep(1)  # Simulate heavy work
        return data * 2

    queue = JobQueue()
    queue.add_job(1, 10)
    queue.add_job(2, 20)

    def worker(job_id):
        print(f"Starting job {job_id}")
        queue.process_job(job_id, slow_processor)
        print(f"Finished job {job_id}")

    # Start two threads processing different jobs
    t1 = threading.Thread(target=worker, args=(1,))
    t2 = threading.Thread(target=worker, args=(2,))

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    print(f"Result 1: {queue.get_result(1)}")
    print(f"Result 2: {queue.get_result(2)}")