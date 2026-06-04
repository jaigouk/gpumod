max_retries = 3
        data = self.jobs.get(job_id)
        if not data: return False

        for attempt in range(max_retries + 1):
            try:
                processor(data)
                return True
            except Exception as e:
                if attempt < max_retries:
                    # Calculate delay: 1, 2, 4
                    delay = 2 ** attempt
                    # Since we aren't actually sleeping per requirements "can be simulated"
                    # and "delays can be stored/tracked rather than actually sleeping"
                    # I'll just keep track or simulate the logic.
                    self.retry_counts[job_id] = attempt + 1
                    # Log/Simulate delay here
                else:
                    return False