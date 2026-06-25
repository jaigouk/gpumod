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
        # 1. Atomically check if job exists and remove it from the pending queue.
        # This prevents multiple threads from picking up the same job.
        with self.lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id)

        # 2. Perform the heavy lifting outside the lock.
        # This allows other threads to call add_job or process_job 
        # while this thread is working.
        try:
            result = processor(data)
        except Exception as e:
            # In a production environment, you might want to store the error 
            # in self.results instead of just letting it crash.
            raise e

        # 3. Atomically store the result.
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)

# --- Example Usage / Test ---
if __name__ == "__main__":
    import time

    queue = JobQueue()

    def slow_processor(data):
        time.sleep(1)  # Simulate a heavy task
        return data * 2

    # Add a job
    queue.add_job("task1", 21)

    # Simulate multiple threads trying to process the same job
    def worker(q, j_id):
        success = q.process_job(j_id, slow_processor)
        print(f"Thread {threading.current_thread().name} success: {success}")

    threads = []
    for i in range(5):
        t = threading.Thread(target=worker, args=(queue, "task1"), name=f"Worker-{i}")
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print(f"Final Result: {queue.get_result('task1')}")