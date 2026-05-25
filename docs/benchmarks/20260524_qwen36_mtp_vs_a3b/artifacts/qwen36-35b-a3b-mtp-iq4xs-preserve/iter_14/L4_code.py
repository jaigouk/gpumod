with self.lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id) # pop is atomic check-and-remove
            # Note: We must process OUTSIDE the lock if the processor is slow,
            # but for a simple fix, we can keep it inside or handle the race strictly.
            # However, the prompt highlights the write race.