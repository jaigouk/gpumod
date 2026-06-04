class JobQueue:
        def __init__(self):
            self.jobs = {}
            self.results = {}

        def add_job(self, job_id, data):
            self.jobs[job_id] = data
            return job_id

        def process_job(self, job_id, Processor):
            if job_id not in self.jobs:
                return False

            data = self.jobs[job_id]
            result = processor(data)

            # BUG: Race condition here
            self.results[job_id] = result
            del self.jobs[job_id]
            return True

        def get_result(self, job_id):
            return self.results.get(job_id)