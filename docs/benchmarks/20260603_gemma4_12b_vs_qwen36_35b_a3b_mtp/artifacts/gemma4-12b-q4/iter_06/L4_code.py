import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # A lock to synchronize access to the internal dictionaries
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, Processor):
        # Phase 1: Safely "claim" the job from the queue
        with self.lock:
            if job_id not in self.jobs:
                return False

            # .pop() removes the item and returns it in one atomic step
            # This prevents two threads from processing the same job_id
            data = self.jobs.pop(job_id)

        # Phase 2: Process the data 
        # This is done OUTSIDE the lock so other threads aren't blocked
        result = Processor(data)

        # Phase 3: Store the result
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)

# Example Usage:
if __name__ == "__main__":
    def slow_processor(data):
        import time
        time.sleep(1) # Simulate work
        return f"Processed: {data}"

    queue = JobQueue()
    queue.add_job("job1", "Payload A")

    # This will now safely handle multiple threads trying to access the same ID
    # or different IDs simultaneously.
    thread1 = threading.Thread(target=queue.process_job, args=("job1", slow_processor))
    thread1.start()
    thread1.join()

    print(queue.get_result("job1"))