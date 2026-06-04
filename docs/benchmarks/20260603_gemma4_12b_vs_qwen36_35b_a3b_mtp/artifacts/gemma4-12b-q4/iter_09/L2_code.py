retries = 0
        while retries <= 3:
            try:
                processor(data)
                return True
            except Exception:
                retries += 1
                if retries > 3: return False
                # simulate delay/log