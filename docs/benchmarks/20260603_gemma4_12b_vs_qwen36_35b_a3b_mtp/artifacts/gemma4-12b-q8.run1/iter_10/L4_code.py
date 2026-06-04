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
            data = None
            with self.lock:
                # .pop() is atomic-ish but inside the lock ensures
                # no other thread can see it or process it.
                data = self.jobs.pop(job_id, None)

            if data is None:
                return False

            # Process the job OUTSIDE the lock so other jobs
            # can be processed concurrently.
            result = processor(data)

            with self.lock:
                self.results[job_id] = result
            return True

        def get_result(self, job_id):
            with self.lock:
                return self.results.get(job_id)