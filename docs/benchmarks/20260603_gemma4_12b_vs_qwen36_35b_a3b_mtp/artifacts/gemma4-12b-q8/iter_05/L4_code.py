import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to synchronize access to the dictionaries
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        with self._lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        data = None

        # Step 1: Atomically check and remove the job from the queue
        # This ensures that only one thread can "claim" a specific job_id
        with self._lock:
            if job_id not in self.jobs:
                return False
            # .pop() removes the item and returns it in one atomic step
            data = self.jobs.pop(job_id)

        # Step 2: Run the processor outside of the lock.
        # This allows multiple jobs to be processed in parallel by different threads.
        try:
            result = processor(data)
        except Exception as e:
            # Handle or log exception based on requirements
            print(f"Error processing job {job_id}: {e}")
            return False

        # Step 3: Atomically store the result
        with self._lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self._lock:
            return self.results.get(job_id)

# Example Usage:
if __name__ == "__main__":
    import time

    queue = JobQueue()
    queue.add_job("job1", "Some Data")

    def slow_processor(data):
        time.sleep(1) # Simulate heavy work
        return f"Processed {data}"

    # This would have crashed or caused race conditions without the locks
    success = queue.process_job("job1", slow_processor)
    print(f"Success: {success}, Result: {queue.get_result('job1')}")