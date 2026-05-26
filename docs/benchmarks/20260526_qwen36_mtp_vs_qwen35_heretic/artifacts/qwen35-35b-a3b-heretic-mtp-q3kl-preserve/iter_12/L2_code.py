for i in range(max_retries):
            try:
                return processor(...)
            except:
                if i < max_retries - 1:
                     wait = 2**i
                     track_backoff(wait)
        return False