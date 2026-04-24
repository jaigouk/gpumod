import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize the lock to protect shared resources
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        # Acquire lock before modifying self.jobs
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # Step 1: Check for job and retrieve data safely
        data = None
        with self.lock:
            if job_id not in self.jobs:
                return False
            # Retrieve the data while holding the lock
            data = self.jobs[job_id]
            # IMPORTANT: Remove the job from the queue immediately to prevent
            # other threads from processing it simultaneously.
            del self.jobs[job_id]

        # Step 2: Execute the external processor (outside the lock)
        # This prevents blocking the entire queue while a long computation runs.
        result = processor(data)

        # Step 3: Store the result safely
        with self.lock:
            self.results[job_id] = result
        
        return True

    def get_result(self, job_id):
        # Acquire lock before reading self.results
        with self.lock:
            return self.results.get(job_id)

# --- Example Usage (Demonstration) ---

def dummy_processor(data):
    import time
    time.sleep(0.01) # Simulate work
    return f"Processed: {data}"

if __name__ == '__main__':
    queue = JobQueue()
    
    # Add jobs
    job_id_1 = queue.add_job(1, "Task A")
    job_id_2 = queue.add_job(2, "Task B")
    
    # Setup threads
    def worker(job_id, processor):
        success = queue.process_job(job_id, processor)
        if success:
            print(f"Thread finished processing job {job_id}")

    t1 = threading.Thread(target=worker, args=(job_id_1, dummy_processor))
    t2 = threading.Thread(target=worker, args=(job_id_2, dummy_processor))

    # Start threads
    t1.start()
    t2.start()

    # Wait for threads to complete
    t1.join()
    t2.join()

    # Check results
    print("\n--- Results ---")
    print(f"Result for Job 1: {queue.get_result(job_id_1)}")
    print(f"Result for Job 2: {queue.get_result(job_id_2)}")
    print(f"Jobs remaining in queue: {queue.jobs}") # Should be empty