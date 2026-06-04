import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # A lock to synchronize access to the dictionaries
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        data = None

        # Step 1: Atomically check and "claim" the job
        with self.lock:
            if job_id not in self.jobs:
                return False
            # pop() removes the item and returns it, 
            # ensuring no other thread can process it.
            data = self.jobs.pop(job_id)

        # Step 2: Process the job outside the lock.
        # This allows multiple threads to run 'processor' simultaneously.
        result = processor(data)

        # Step 3: Atomically write the result
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
    queue.add_job("job1", "Some Data")

    def slow_processor(data):
        time.sleep(1) # Simulate heavy work
        return f"Processed: {data}"

    # This would have caused a race condition in the original code
    # if multiple threads called process_job("job1", ...) simultaneously.
    t1 = threading.Thread(target=queue.process_job, args=("job1", slow_processor))
    t2 = threading.Thread(target=queue.process_job, args=("job1", slow_processor))

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    print(f"Result: {queue.get_result('job1')}")