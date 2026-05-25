def process_job(self, job_id, processor):
            with self.lock:
                if job_id not in self.jobs:
                    return False

                # "Pop" the job to claim it.
                # This prevents other threads from picking it up.
                data = self.jobs.pop(job_id)

            # Process outside the lock to allow concurrency for other jobs
            result = processor(data)

            with self.lock:
                self.results[job_id] = result
                # Note: We already popped it, so no need to del here,
                # but the original code had del. If we pop, we are safe.
                # However, if we want to stick strictly to the original logic structure:
                # del self.jobs[job_id] was inside the lock in the original (implicitly, though unguarded).