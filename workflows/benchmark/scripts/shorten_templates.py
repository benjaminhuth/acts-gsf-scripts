#!/bin/python3

import sys

def shorten(line, brace_pair):
    stack = []
    pairs = []

    outer_pairs = []

    for i, char in enumerate(line):
        if char == brace_pair[0]:
            stack.append(i+1)
        if char == brace_pair[1]:
            try:
                pairs.append((stack.pop(), i))
            except:
                return line

        if len(stack) == 0 and len(pairs) > 0:
            outer_pairs.append(pairs[-1])

    outer_pairs = list(set(outer_pairs))
    outer_pairs.sort(key=lambda p: p[0])

    removed = 0

    for i, (b,e) in enumerate(outer_pairs):
        line = line[:b-removed] + "…" + line[e-removed:]
        removed += e-b-1

    return line


if __name__ == "__main__":
    for line in sys.stdin.readlines():
        line = shorten(line, ("<", ">"))
        line = shorten(line, ("(", ")"))
        
        # replace these as this is actually an operator name and not real parentheses
        line = line.replace("operator(…)", "operator()")
        
        sys.stdout.write(line)
