with self.lock:
            if job_id not in self.jobs:
                return False # Or raise exception
            data = self.jobs[job_id]
            del self.jobs[job_id] # Claim the job
        # Process outside lock?
        # If we process inside the lock, we block other threads from adding/processing jobs.
        # If we process outside, we need to be careful.