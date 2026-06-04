import threading

    class JobQueue:
        def __init__(self):
            self.jobs = {}
            self.results = {}
            self.lock = threading.Lock()

        def add_job(self, job_id, data):
            with self.lock:
                self.jobs[job_id] = data
            return job_id

        def process_job(self, job_id, processor):
            # We use a lock to ensure that checking and removing the job
            # from the queue is an atomic operation.
            with self.lock:
                if job_id not in self.jobs:
                    return False
                # .pop() removes it immediately so no other thread
                # can pick it up while processing.
                data = self.jobs.pop(job_id)

            # Processing happens outside the lock to allow parallelism
            result = processor(data)

            with self.lock:
                self.results[job_id] = result
            return True

        def get_result(self, job_id):
            with self.lock:
                return self.results.get(job_id)