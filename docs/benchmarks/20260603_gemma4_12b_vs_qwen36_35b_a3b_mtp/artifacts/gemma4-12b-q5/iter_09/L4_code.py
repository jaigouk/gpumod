def process_job(self, job_id, Processor):
        if job_id not in self.jobs: # Race: another thread might delete it here
            return False

        data = self.jobs[job_id] # Race: another thread might delete it here
        result = processor(data)

        self.results[job_id] = result # Race: multiple threads writing same key
        del self.jobs[job_id] # Race: Multiple threads trying to del same key
        return True