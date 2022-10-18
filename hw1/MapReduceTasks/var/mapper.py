#!/usr/bin/env python

import sys


COLUMN_POS = 9


def main(file=sys.stdin):
    summer = 0
    count = 0
    vals = []
    for line in file:
        try:
            price = line.strip().split(",")[COLUMN_POS]
            price = float(price)
            summer += price
            count += 1
            vals.append(price)
        except Exception:
            continue
    mean = summer / count
    var = sum([(x - mean) ** 2 for x in vals]) / (count - 1)
    print("{}\t{}\t{}".format(count, mean, var))


if __name__ == "__main__":
    main()