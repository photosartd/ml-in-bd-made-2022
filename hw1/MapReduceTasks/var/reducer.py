#!/usr/bin/env python

import sys


def main(file=sys.stdin):
    m_curr = 0
    c_curr = 0
    v_curr = 0
    for line in file:
        try:
            c_new, m_new, v_new = map(float, line.strip().split("\t"))
            m_curr = (c_curr * m_curr + c_new * m_new) / (c_curr + c_new)
            v_curr = (c_curr * v_curr + c_new * v_new) / (c_curr + c_new) + c_curr * c_new * ((m_curr - m_new) / (c_curr + c_new)) ** 2
            c_curr += c_new
        except Exception:
            continue
    print("Var: {}".format(v_curr))


if __name__ == "__main__":
    main()