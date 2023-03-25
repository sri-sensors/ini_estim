import ini_estim.config as config
import argparse
import sys


def list_current():
    cfg = config.get_config()
    for k, v in cfg.items():
        print(k, v, sep="\t")


def update(name, value):
    cfg = config.get_config()
    if name not in cfg:
        print("Invalid configuration option {}!".format(name), file=sys.stderr)
        exit(-1)
    old_value = cfg[name]
    if old_value is not None:
        value = type(old_value)(value)
    cfg[name] = value
    config.save_config(cfg)


def parse_args(args=None, prog=None):
    parser = argparse.ArgumentParser(
        prog=prog, description="Configure ini_estim."
    )
    sub = parser.add_subparsers(
        title="command", description="configuration command", dest="command")
    sub.add_parser("list", help="List current configuration")
    p = sub.add_parser("update", help="Update value")
    p.add_argument("name", type=str, help="Name of setting")
    p.add_argument("value", help="New value")

    args = parser.parse_args(args)
    if args.command is None:
        args.command = "list"
    return args

def main(args=None, prog=None):
    args = parse_args(args=args, prog=prog)
    if args.command == "list":
        list_current()
    elif args.command == "update":
        update(args.name, args.value)
    else:
        print("Unknown command.", file=sys.stderr)

if __name__ == "__main__":
    main()
