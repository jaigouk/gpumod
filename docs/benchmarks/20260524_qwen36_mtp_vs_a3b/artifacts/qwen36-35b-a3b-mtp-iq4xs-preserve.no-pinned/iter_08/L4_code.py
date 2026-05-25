def process_job(self, job_id, processor):
            with self.lock:
                if job_id not in self.jobs:
                    return False

                data = self.jobs[job_id]
                del self.jobs[job_id] # Move deletion inside lock to prevent double processing

            # Processing happens outside lock?
            # If processor is slow, holding the lock is bad.
            # However, if we release the lock before processing, another thread could theoretically
            # try to process the same job_id if we didn't delete it.
            # But we deleted it above. So it's safe to process outside?
            # Wait, if we delete it, no one else can pick it up.
            # But what about the result? self.results is shared.
            # Writing to self.results should probably be locked too, or at least safe.