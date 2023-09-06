import os


class FileLock:
    def __init__(self, lock_file):
        self.lock_file = f"{lock_file}.lock"
        self.acquired = False

    def acquire(self):
        """Try to acquire the lock. Returns True if successful, False otherwise."""
        try:
            # Open the lock file in write mode
            with open(self.lock_file, "x") as f:
                f.write("locked")
            # Lock file was successfully created, lock is acquired
            self.acquired = True
            return True
        except FileExistsError:
            # Lock file already exists, lock acquisition failed
            return False

    def release(self):
        """Release the lock."""
        if self.acquired:
            try:
                os.remove(self.lock_file)
            except FileNotFoundError:
                raise Exception(f"Lock file {self.lock_file} is already removed!")

    def get_lock_file(self):
        return self.lock_file

    def __enter__(self):
        return self.acquire()

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
