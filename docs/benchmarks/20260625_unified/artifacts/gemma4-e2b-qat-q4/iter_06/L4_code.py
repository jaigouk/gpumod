import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # 1. Initialize a lock to guard shared resources
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        # 2. Protect the write operation to self.jobs
        with self.lock:
            self.jobs[job_id] = data
            return job_id

    def process_job(self, job_id, processor):
        # 3. Protect the entire workflow: checking jobs, reading jobs, 
        # writing results, and deleting jobs.
        with self.lock:
            if job_id not in self.jobs:
                return False

            # Retrieve data
            data = self.jobs[job_id]

            # Execute the processor (Note: For extremely long processing tasks, 
            # consider performing the heavy computation outside the lock, 
            # but keeping it inside ensures thread safety for the queue management.)
            result = processor(data)

            # Write result
            self.results[job_id] = result

            # Remove job
            del self.jobs[job_id]
            return True

    def get_result(self, job_id):
        # 4. Protect the read operation on self.results
        with self.lock:
            return self.results.get(job_id)

# Example Usage (Optional, for testing concurrency):
if __name__ == '__main__':
    queue = JobQueue()

    def mock_processor(data):
        import time
        time.sleep(0.1) # Simulate work
        return f"Processed: {data}"

    job_id_1 = queue.add_job("J1", "DataA")
    job_id_2 = queue.add_job("J2", "DataB")

    # Start two threads processing jobs
    thread1 = threading.Thread(target=queue.process_job, args=(job_id_1, mock_processor))
    thread2 = threading.Thread(target=queue.process_job, args=(job_id_2, mock_processor))

    thread1.start()
    thread2.start()

    # Wait for both threads to finish
    thread1.join()
    thread2.join()

    print(f"Result for J1: {queue.get_result('J1')}")
    print(f"Result for J2: {queue.get_result('J2')}")
    print(f"Result for J3 (Non-existent): {queue.get_result('J3')}")