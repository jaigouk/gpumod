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
        # 1. Atomically check and remove the job from the queue
        # This prevents multiple threads from picking up the same job.
        with self.lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id)

        # 2. Execute the heavy processing OUTSIDE the lock.
        # This allows other threads to add jobs or process other jobs 
        # while this thread is busy computing.
        try:
            result = processor(data)
            success = True
        except Exception as e:
            result = e
            success = False

        # 3. Atomically store the result
        with self.lock:
            self.results[job_id] = result

        return success

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)

# --- Example Usage / Test ---
if __name__ == "__main__":
    import time

    def slow_processor(data):
        time.sleep(1)  # Simulate heavy work
        return data * 2

    queue = JobQueue()

    # Adding jobs
    queue.add_job(1, 10)
    queue.add_job(2, 20)

    # Simulating multiple worker threads
    def worker(q, job_id, proc):
        print(f"Starting job {job_id}")
        q.process_job(job_id, proc)
        print(f"Finished job {job_id}")

    t1 = threading.Thread(target=worker, args=(queue, 1, slow_processor))
    t2 = threading.Thread(target=worker, args=(queue, 2, slow_processor))

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    print(f"Results: {queue.get_result(1)}, {queue.get_result(2)}")