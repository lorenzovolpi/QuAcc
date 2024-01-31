import argparse

from quacc.main import main as run_local
from remote import remote as run_remote


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--local", action="store_true", dest="local")
    parser.add_argument("-r", "--remote", action="store_true", dest="remote")
    parser.add_argument("-d", "--detatch", action="store_true", dest="detatch")
    args = parser.parse_args()

    if args.local:
        run_local()
    elif args.remote:
        run_remote(detatch=args.detatch)


if __name__ == "__main__":
    run()
