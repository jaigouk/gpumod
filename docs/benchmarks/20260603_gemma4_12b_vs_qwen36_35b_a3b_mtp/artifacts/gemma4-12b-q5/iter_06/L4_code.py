import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # A lock to synchronize access to the jobs and results dictionaries
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, Processor):
        data = None

        # Phase 1: Atomically check and "claim" the job
        with self.lock:
            if job_id not in self.jobs:
                return False

            # We use .pop() to remove the job from the queue immediately.
            # This ensures that no other thread can pick up this same job_id.
            data = self.jobs.pop(job_id)

        # Phase 2: Run the processor outside of the lock.
        # This allows multiple jobs to be processed in parallel.
        try:
            result = Processor(data)
        except Exception as e:
            # Optional: Handle errors so the job doesn't just vanish
            print(f"Error processing job {job_id}: {e}")
            return False

        # Phase 3: Atomically store the result
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)

# Example Usage:
if __name__ == "__main__":
    import time

    def slow_processor(data):
        time.sleep(1)  # Simulate heavy work
        return f"Processed: {data}"

    queue = JobQueue()
    queue.add_job("task1", "Payload A")

    # Simulate concurrent access
    def worker():
        success = queue.process_job("task1", slow_processor)
        print(f"Worker success: {success}")

    thread1 = threading.Thread(target=worker)
    thread2 = threading.Thread(target=worker)

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()

    print(f"Final Result: {queue.get_result('task1')}")