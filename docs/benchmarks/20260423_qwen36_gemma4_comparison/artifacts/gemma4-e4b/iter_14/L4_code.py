import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize the lock to protect shared state (self.jobs and self.results)
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        """Adds a job to the queue."""
        with self._lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        """
        Retrieves a job, processes it, and stores the result atomically.
        """
        data = None
        
        # Step 1: Atomically check for and retrieve the job data
        with self._lock:
            if job_id not in self.jobs:
                return False
            # Copy the data so the processor runs outside the lock 
            # (to avoid blocking the queue for the duration of the computation)
            data = self.jobs[job_id]
            # Immediately remove the job from the queue while holding the lock
            del self.jobs[job_id]

        # Step 2: Process the job (this happens outside the lock)
        if data is None:
             # Should not happen if logic is correct, but good practice
             return False
             
        result = processor(data)

        # Step 3: Atomically store the result
        with self._lock:
            self.results[job_id] = result
            
        return True

    def get_result(self, job_id):
        """Retrieves a result safely."""
        with self._lock:
            return self.results.get(job_id)

# Example Usage (Demonstration of safety)
if __name__ == '__main__':
    queue = JobQueue()
    
    # Simple processor function
    def simple_processor(data):
        import time
        time.sleep(0.01) # Simulate work
        return f"Processed: {data}"

    job_id_1 = queue.add_job(1, "Task A")
    job_id_2 = queue.add_job(2, "Task B")

    def worker(job_id, queue_instance, processor):
        if queue_instance.process_job(job_id, processor):
            print(f"Worker finished job {job_id}")
        else:
            print(f"Worker failed to find job {job_id}")

    t1 = threading.Thread(target=worker, args=(job_id_1, queue, simple_processor))
    t2 = threading.Thread(target=worker, args=(job_id_2, queue, simple_processor))

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    print("\n--- Results ---")
    print(f"Result 1: {queue.get_result(job_id_1)}")
    print(f"Result 2: {queue.get_result(job_id_2)}")
    print(f"Jobs remaining in queue: {len(queue.jobs)}") # Should be 0