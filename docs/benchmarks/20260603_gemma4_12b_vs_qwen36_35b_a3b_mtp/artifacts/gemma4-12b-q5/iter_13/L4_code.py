import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to synchronize access to the dictionaries
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        data = None

        # Phase 1: Atomic "Check and Remove"
        # We use the lock only to retrieve the data and remove it 
        # from the jobs dict so no other thread can process it.
        with self.lock:
            if job_id not in self.jobs:
                return False

            # .pop() retrieves the value and removes the key in one step
            data = self.jobs.pop(job_id)

        # Phase 2: Processing
        # We release the lock here so other threads can add/process 
        # other jobs simultaneously while this thread works.
        result = processor(data)

        # Phase 3: Atomic "Write Result"
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)