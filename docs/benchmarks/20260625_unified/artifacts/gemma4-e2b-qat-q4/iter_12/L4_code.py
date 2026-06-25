import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to synchronize access to the shared resources
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        # Acquire the lock before modifying the shared 'jobs' dictionary
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # Acquire the lock to ensure atomic check, read, and deletion/write operations
        with self.lock:
            if job_id not in self.jobs:
                return False

            # 1. Retrieve data (read from jobs)
            data = self.jobs[job_id]

            # Note: We perform the processing outside the lock if the processor is long-running 
            # to prevent blocking all other queue operations. However, the critical steps 
            # (retrieving data and updating results) must remain protected.

        # --- Processing outside the lock (optional, but good practice for heavy work) ---
        result = processor(data)

        # Re-acquire the lock to update results and clean up the job queue
        with self.lock:
            # 2. Store the result
            self.results[job_id] = result

            # 3. Remove the job
            del self.jobs[job_id]

        return True

    def get_result(self, job_id):
        # Acquire the lock when reading the 'results' dictionary
        with self.lock:
            return self.results.get(job_id)

# Example Usage (for demonstration):
if __name__ == '__main__':
    queue = JobQueue()

    def dummy_processor(data):
        # Simulate some work
        import time
        time.sleep(0.1)
        return f"Processed: {data}"

    queue.add_job("J1", "Data for job 1")
    queue.add_job("J2", "Data for job 2")

    # Simulate concurrent processing
    thread1 = threading.Thread(target=queue.process_job, args=("J1", dummy_processor))
    thread2 = threading.Thread(target=queue.process_job, args=("J2", dummy_processor))

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()

    print("Result J1:", queue.get_result("J1"))
    print("Result J2:", queue.get_result("J2"))
    print("Result J3:", queue.get_result("J3")) # Should be None