import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to synchronize access to the dictionaries
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        data = None

        # Phase 1: Atomically "claim" the job.
        # We pop the job from the dict so only one thread can successfully 
        # retrieve the data.
        with self.lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id)

        # Phase 2: Execute the processor.
        # We do this OUTSIDE the lock. This allows multiple threads to 
        # run different processors concurrently.
        result = processor(data)

        # Phase 3: Store the result.
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)

# Example Usage:
if __name__ == "__main__":
    import time

    queue = JobQueue()
    queue.add_job("task1", {"value": 10})

    def slow_processor(data):
        time.sleep(1)  # Simulate heavy work
        return data["value"] * 2

    # Simulate two threads trying to grab the same job
    t1 = threading.Thread(target=queue.process_job, args=("task1", slow_processor))
    t2 = threading.Thread(target=queue.process_job, args=("task1", slow_processor))

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    print(f"Result: {queue.get_result('task1')}")