#!/bin/python3

import sys

if __name__ == "__main__":
    for line in sys.stdin.readlines():
        line = line.replace("Acts::detail::", "")
        line = line.replace("Acts::", "")
        line = line.replace("boost::container::", "")
        line = line.replace("boost::", "")
        sys.stdout.write(line)
