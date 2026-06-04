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

        # Phase 1: Atomically check if job exists and "claim" it
        with self.lock:
            if job_id not in self.jobs:
                return False

            # Using .pop() ensures that only ONE thread can successfully
            # retrieve the data and remove the job from the queue.
            data = self.jobs.pop(job_id)

        # Phase 2: Execute the processor (OUTSIDE the lock)
        # This allows multiple threads to run different processors simultaneously.
        result = processor(data)

        # Phase 3: Atomically save the result
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)

# Example usage:
if __name__ == "__main__":
    import time

    queue = JobQueue()
    queue.add_job("job1", 10)
    queue.add_job("job2", 20)

    def slow_processor(n):
        time.sleep(1) # Simulate heavy work
        return n * 2

    # Simulate concurrent processing
    t1 = threading.Thread(target=queue.process_job, args=("job1", slow_processor))
    t2 = threading.Thread(target=queue.process_job, args=("job2", slow_processor))

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    print(f"Result 1: {queue.get_result('job1')}")
    print(f"Result 2: {queue.get_result('job2')}")