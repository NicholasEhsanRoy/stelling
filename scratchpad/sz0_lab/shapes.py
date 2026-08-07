# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Q1: can an `and` output lose elements of an operand for any reason
other than a zero-size output?
Q2: can an `and` node be size-0 while an ANCESTOR `and` is not?"""
import itertools, math
import numpy as np

SHAPES = [(), (1,), (0,), (2,), (3,), (1,1), (2,1), (1,3), (2,3), (2,0),
          (0,3), (0,), (1,0), (0,0), (2,1,3), (2,0,3)]

def bshape(a, b):
    try:
        return np.broadcast_shapes(a, b)
    except ValueError:
        return None

def size(s):
    return math.prod(s) if s else 1

lossy = []
for a, b in itertools.product(SHAPES, repeat=2):
    out = bshape(a, b)
    if out is None:
        continue
    # does every element of `a` appear at least once in the broadcast output?
    A = np.arange(size(a)).reshape(a)
    seen = set(np.broadcast_to(A, out).ravel().tolist())
    if len(seen) != size(a):
        lossy.append((a, b, out, size(a), size(out)))

print("LOSSY (an operand element vanishes) pairs:", len(lossy))
allzero = all(size(o) == 0 for _a, _b, o, _sa, _so in lossy)
print("every lossy pair has a ZERO-SIZE output:", allzero)
nonzero = [r for r in lossy if size(r[2]) != 0]
print("counterexamples:", nonzero[:5])

# Q2: nonzero output from an operand of size 0
bad = [(a, b, bshape(a, b)) for a, b in itertools.product(SHAPES, repeat=2)
       if bshape(a, b) is not None and size(a) == 0 and size(bshape(a, b)) != 0]
print("size-0 operand yielding a NONZERO broadcast output:", bad)
