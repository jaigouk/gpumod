import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to synchronize access to the shared dictionaries
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        data = None

        # Phase 1: Claim the job
        # We use a lock to ensure only one thread can 'pop' a specific job_id
        with self.lock:
            if job_id not1 in self.jobs:
                return False

            # .pop() removes the item and returns it in one atomic step
            # This prevents other threads from picking up the same job.
            data = self.jobs.pop(job_id)

        # Phase 2: Execute the work
        # We do NOT hold the lock here so that multiple threads can 
        # run the processor() function in parallel.
        try:
            result = processor(data)
        except Exception as e:
            # Optional: Handle errors here (e.g., log them or put job back)
            raise e

        # Phase 3: Store the result
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
    queue.add_job("task1", 10)
    queue.add_job("task2", 20)

    def slow_processor(n):
        time.sleep(1) # Simulate heavy work
        return n * 2

    # Simulate multiple threads trying to process the same job
    t1 = threading.Thread(target=queue.process_job, args=("task1", slow_processor))
    t2 = threading.Thread(target=queue.process_job, args=("task1", slow_processor))
    t3 = threading.Thread(target=queue.process_job, args=("task2", slow_processor))

    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()

    print(f"Result 1: {queue.get_result('task1')}") # Should be 20
    print(f"Result 2: {queue.get_result('task2')}") # Should be 40