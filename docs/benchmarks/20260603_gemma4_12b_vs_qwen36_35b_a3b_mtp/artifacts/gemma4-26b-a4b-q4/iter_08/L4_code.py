import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # 1. Attempt to "claim" the job atomically
        with self.lock:
            if job_id not in self.jobs:
                return False
            # .pop() removes the job immediately so no other 
            # thread can claim it while we are processing
            data = self.jobs.pop(job_id)

        # 2. Run the processor OUTSIDE the lock.
        # This allows other threads to add/process other jobs 
        # while this one is doing heavy lifting.
        result = processor(data)

        # 3. Re-acquire lock to safely write the result
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)

# --- Testing the fix ---
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

    # Start multiple threads trying to process the same jobs
    threads = []
    for _ in range(4):
        threads.append(threading.Thread(target=worker, args=(1,)))
        threads.append(threading.Thread(target=worker, args=(2,)))

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"Result for job 1: {queue.get_result(1)}")
    print(f"Result for job 2: {queue.get_result(2)}")