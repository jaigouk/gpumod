import threading
        class JobQueue:
            def __init__(self):
                self.jobs = {}
                self.results = {}
                self.lock = threading.Lock() # Or RLock if re-entrant needed, but Lock is sufficient here.