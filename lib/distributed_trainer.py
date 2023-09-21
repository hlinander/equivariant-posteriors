import filelock
import time

from lib.ddp import ddp_setup
from lib.train import load_or_create_state
from lib.train import do_training
from lib.stable_hash import stable_hash_small

from lib.train_distributed import fetch_requested_train_run, report_done


def do_train_run(train_run, device_id):
    state = load_or_create_state(train_run, device_id)
    do_training(train_run, state, device_id)


def main():
    device_id = ddp_setup()
    while True:
        print("Trying to fetch train run...")
        distributed_train_run = fetch_requested_train_run()
        if distributed_train_run is not None:
            try:
                do_train_run(distributed_train_run.train_run, device_id)
                report_done(distributed_train_run)
            except filelock.Timeout:
                print(
                    f"Couldn't aquire training lock for {stable_hash_small(distributed_train_run.train_run.train_config)}"
                )
            finally:
                distributed_train_run.lock.release()
        time.sleep(1)


if __name__ == "__main__":
    main()
