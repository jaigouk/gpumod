import threading

        class JobQueue:
            def __init__(self):
                self.jobs = {}
                self.results = {}
                self.lock = threading.Lock() # Add lock

            def add_job(self, job_id, data):
                with self.lock: # Thread-safe insertion
                    self.jobs[job_id] = data
                return job_id

            def process_job(self, job_id, processor):
                with self.lock: # Lock for check and delete
                    if job_id not in self.jobs:
                        return False
                    
                    # It is safer to pop here to prevent double processing, 
                    # but the original code kept it in jobs until the end.
                    # Let's stick close to original logic but make it safe.
                    # Actually, if we don't remove it inside the lock, 
                    # another thread could grab it right after we release the lock 
                    # but before we process? 
                    # No, because processing happens outside the lock in a real app, 
                    # but here processing is inside the method.
                    
                    # To fix the race condition strictly:
                    # We need to ensure that the check, the retrieval, and the 
                    # removal are atomic, OR that the processing is protected.
                    
                    # If we process INSIDE the lock:
                    # self.lock.acquire()
                    # if job_id in jobs:
                    #    data = jobs.pop(job_id)
                    #    result = processor(data)
                    #    results[job_id] = result
                    # self.lock.release()
                    # This serializes all processing.
                    
                    # If we process OUTSIDE the lock:
                    # self.lock.acquire()
                    # if job_id in jobs:
                    #    data = jobs.pop(job_id) # Remove immediately
                    # else: return False
                    # self.lock.release()
                    # result = processor(data)
                    # self.lock.acquire()
                    # results[job_id] = result
                    # self.lock.release()
                    
                    # The original code does: Check -> Get -> Process -> Write -> Delete.
                    # The bug is that Check and Delete are separated by Process.
                    # If Thread A is processing, Thread B checks and sees job exists.
                    # Thread A finishes, deletes.
                    # Thread B processes (redundant work).
                    # Thread B deletes (KeyError? No, it has the data variable, 
                    # but tries to del self.jobs[job_id]).
                    # Wait, del self.jobs[job_id] will raise KeyError if Thread A already deleted it.
                    
                    # So the bug is actually a KeyError on delete OR redundant processing.
                    
                    # Let's fix it by locking the critical sections.
                    pass