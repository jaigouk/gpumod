import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize the lock to protect shared resources
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        # Acquire lock before modifying shared state
        with self._lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        result = None
        success = False
        
        # Acquire lock to ensure atomic read/write/delete sequence
        with self._lock:
            if job_id not in self.jobs:
                return False

            # 1. Get data and remove job immediately (critical step)
            data = self.jobs[job_id]
            del self.jobs[job_id]
            
            # 2. Process the job (NOTE: We release the lock here for performance 
            # if the processor function is long-running. If the processor must 
            # access other shared state, the lock must be held.)
            # For this simple example, we'll run the processing outside the lock 
            # to prevent holding the lock during potentially long CPU work.
            pass # Data is retrieved, job is removed.
        
        # Run the CPU-intensive task outside the lock
        try:
            result = processor(data)
            success = True
        except Exception as e:
            print(f"Error processing job {job_id}: {e}")
            # Optionally, handle failed jobs here
            return False


        # Acquire lock again to safely write the result
        with self._lock:
            if success:
                self.results[job_id] = result
            # If processing failed, we might choose to leave the job out of results
            
        return success

    def get_result(self, job_id):
        # Acquire lock even for reading to ensure we don't read results 
        # while another thread is concurrently deleting or updating the dictionary.
        with self._lock:
            return self.results.get(job_id)

# --- Example Usage Verification ---

def mock_processor(data):
    # Simulate work
    import time
    time.sleep(0.01) 
    return f"Processed_{data}"

if __name__ == '__main__':
    queue = JobQueue()
    
    # Add jobs
    job_id1 = queue.add_job(1, "Task A")
    job_id2 = queue.add_job(2, "Task B")
    
    threads = []
    
    def worker(job_id):
        queue.process_job(job_id, mock_processor)

    # Create multiple threads trying to process the jobs
    t1 = threading.Thread(target=worker, args=(job_id1,))
    t2 = threading.Thread(target=worker, args=(job_id2,))
    
    threads.append(t1)
    threads.append(t2)

    t1.start()
    t2.start()

    for t in threads:
        t.join()

    print(f"Job 1 result: {queue.get_result(job_id1)}")
    print(f"Job 2 result: {queue.get_result(job_id2)}")
    
    # Verify that no jobs are left in the queue
    with queue._lock:
        print(f"Jobs remaining in queue: {queue.jobs}")