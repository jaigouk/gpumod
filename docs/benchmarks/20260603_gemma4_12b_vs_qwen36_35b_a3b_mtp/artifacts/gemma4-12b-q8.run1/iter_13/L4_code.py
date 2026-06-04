if job_id not in self.jobs: # Thread A checks
            return False
        # Thread B checks, also finds it exists
        data = self.jobs[job_id] # Thread A gets data
        result = processor(data) # Thread A runs slow processor
        # Thread B gets same data
        # Thread B runs same processor
        self.results[job_id] = result # Thread A writes
        del self.jobs[job_id] # Thread A deletes
        self.results[job_id] = result # Thread B writes (redundant)
        del self.jobs[job_id] # Thread B fails because it was already deleted!