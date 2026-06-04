import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to synchronize access to shared dictionaries
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        data = None

        # Critical Section 1: Check and "claim" the job
        with self.lock:
            if job_id not in self.jobs:
                return False

            # .pop() removes the item and returns it atomically
            # This ensures no other thread can "grab" this job
            data = self.jobs.pop(job_id)

        # Processing happens OUTSIDE the lock.
        # This allows multiple threads to run 'processor' simultaneously.
        result = processor(data)

        # Critical Section 2: Save the result
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
    queue.add_job("job1", "some data")

    def slow_processor(data):
        time.sleep(random.uniform(0.1, 0.5))
        return f"Processed: {data}"

    # Simulate concurrent access
    def worker(jid):
        success = queue.process_job(jid, slow_processor)
        print(f"Thread {threading.current_thread().name} success: {success}")

    threads = [threading.Thread(target=worker, args=("job1",)) for _ in range(3)]

    for t in threads: t.start()
    for t in threads: t.join()

    print(f"Final Result: {queue.get_result('job1')}")