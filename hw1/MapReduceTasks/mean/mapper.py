#!/usr/bin/env python

import sys


COLUMN_POS = 9


def main(file=sys.stdin):
    summer = 0.0
    count = 0
    for line in file:
        try:
            price = line.strip().split(",")[COLUMN_POS]
            summer += float(price)
            count += 1
        except Exception:
            continue
    mean = summer / count
    print("{}\t{}".format(count, mean))


if __name__ == "__main__":
    main()