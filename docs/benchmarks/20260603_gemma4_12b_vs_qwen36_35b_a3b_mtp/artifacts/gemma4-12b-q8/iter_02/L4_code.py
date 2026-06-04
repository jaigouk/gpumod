import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to synchronize access to shared resources
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        with self.lock:
            self.jobs[job_id] = data
            return job_id

    def process_job(self, job_id, processor):
        # Step 1: Atomically check and remove the job from the queue.
        # This prevents multiple threads from processing the same job.
        with self.lock:
            if job_id not in self.jobs:
                return False
            # .pop() removes the key and returns the value in one step
            data = self.jobs.pop(job_id)

        # Step 2: Process the data outside of the lock.
        # This allows other threads to add/process other jobs concurrently.
        result = processor(data)

        # Step 3: Store the result inside the lock.
        with self.lock:
            self.results[job_id] = result
            return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)

# --- Example Usage ---
if __name__ == "__main__":
    import time
    import random

    queue = JobQueue()

    def slow_processor(data):
        time.sleep(random.uniform(0.1, 0.5))
        return f"Processed {data}"

    # Add a job
    queue.add_job("task1", "some_data")

    # Simulate two threads trying to process the same job at once
    def worker(id):
        success = queue.process_job("task1", slow_processor)
        print(f"Thread {id} success: {success}")

    t1 = threading.Thread(target=worker, args=(1,))
    t2 = threading.Thread(target=worker, args=(2,))

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    print(f"Final Result: {queue.get_result('task1')}")