import argparse


def main():
    parser = argparse.ArgumentParser(description="Model Arena")
    sub = parser.add_subparsers(dest="command", required=True)
    version = sub.add_parser("version")
    version.set_defaults(command="version")
    monitor = sub.add_parser("monitor")
    monitor.add_argument("publication")
    args = parser.parse_args()
    if args.command == "version":
        from . import __version__
        print(__version__)
    elif args.command == "monitor":
        from .observatory import Observatory
        handle = Observatory(args.publication).launch()
        print(handle.url)
        try:
            handle.process.wait()
        except KeyboardInterrupt:
            handle.stop()

