attempts = 0
        max_retries = 3
        while attempts <= max_retries:
            try:
                processor(data)
                return True
            except Exception:
                if attempts == max_retries:
                    return False
                attempts += 1
                delay = 2**(attempts - 1)
                # simulate delay...