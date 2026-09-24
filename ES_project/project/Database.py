# Database

from typing import List, Callable
from collections import deque
import numpy as np
from .cube_state import rubik_cube
import copy

# ========== Factorials for Permutation Encoding ==========
factorial = [1] * 13
for i in range(1, 13):
    factorial[i] = factorial[i - 1] * i

opposite_face = [1, 0, 3, 2, 5, 4]

# ========== Encoding Utilities ==========
def encode_perm(p: List[int], factorial: List[int], N: int, S: int) -> int:
    pos = list(range(N))
    elem = list(range(N))
    v = 0
    for i in range(S):
        t = pos[p[i]]
        v += factorial[i] * t
        pos[elem[N - i - 1]] = t
        elem[t] = elem[N - i - 1]
    return v

def encode_corner_orientation(cube: rubik_cube.cube_t) -> int:
    _, co = cube.get_corner()
    return sum(co[i] * (3**i) for i in range(7))

def encode_edge_orientation(cube: rubik_cube.cube_t) -> int:
    _, eo = cube.get_edge()

    val = 0
    for i in range(11):  # 只取前 11 個位置
        val = (val << 1) | eo[i]
    return val

def encode_slice(cube: rubik_cube.cube_t) -> int:
    ep, _ = cube.get_edge()  
    slice_bits = 0
    for pos in range(12):
        if 8 <= ep[pos] <= 11:  # 中層 slice 的 edge 編號是 8~11
            slice_bits |= (1 << pos)
    return slice_bits

def encode_cp(cube: rubik_cube.cube_t) -> int:
    cp, _ = cube.get_corner()
    perm = cp[:]
    code = 0
    for i in range(7):
        smaller = sum(1 for j in range(i+1, 8) if perm[j] < perm[i])
        code += smaller * factorial[7 - i]
    return code

def encode_ep1(cube):
    ep, _ = cube.get_edge()  # ep[position] = piece
    indices = []
    slice_positions = [8, 9, 10, 11]
    for pos in slice_positions:
        indices.append(ep[pos])
    if len(indices) != 4:
        return -1
    # 將位置順序 encode
    return encode_perm(indices, factorial[:4], 12, 4)

def encode_ep2(cube):
    ep, _ = cube.get_edge()  # ep[position] = piece
    indices = []
    for pos in range(12):
        if pos < 8:
            indices.append(ep[pos])
    if len(indices) != 8:
        return -1
    return encode_perm(indices, factorial[:8], 12, 8)



# ========== BFS Database Builder ==========
def build_pdb(encoder: Callable, size: int, filename: str, max_depth: int = 20, phase2_restriction: bool = False):
    table = np.full(size, -1, dtype=np.int8)
    init_cube = rubik_cube.cube_t()
    init_key = encoder(init_cube)
    table[init_key] = 0

    q = deque()
    q.append((init_cube, 0, -1))

    PHASE2_FACES = [1, 2, 3, 4, 5]      # 0 is not used
    PHASE2_TURNS = {
        1: [1, 2, 3],
        2: [2],
        3: [2],
        4: [2],
        5: [2],
    }

    while q:
        cube, depth, last_face = q.popleft()
        if depth >= max_depth:
            continue

        for face in PHASE2_FACES:
            if face == last_face:
                continue
            if last_face != -1 and face == opposite_face[last_face]:
                continue

            turns = PHASE2_TURNS[face] if phase2_restriction else [1, 2, 3]
            for t in turns:
                new_cube = copy.deepcopy(cube)
                new_cube.rotate(face, t)
                key = encoder(new_cube)
                if key == -1:
                    continue
                if table[key] == -1:
                    table[key] = depth + 1
                    q.append((new_cube, depth + 1, face))

    np.save(filename, table)

# ========== Entry Points ==========
if __name__ == "__main__":
    print("Building CO Table...")
    build_pdb(encode_corner_orientation, 2187, "data/pattern_tables/pdb_co.npy")        # 3^7

    print("Building EO Table...")
    build_pdb(encode_edge_orientation, 2048, "data/pattern_tables/pdb_eo.npy")          # 2^11

    print("Building Slice Table...")
    build_pdb(encode_slice, 4096, "data/pattern_tables/pdb_slice.npy")                  #C(12, 4)

    print("Building CP Table...")
    build_pdb(encode_cp, 40320, "data/pattern_tables/phase2_cp_table.npy", max_depth=20, phase2_restriction=True)       # 8!

    print("Building EP1 Table...")
    build_pdb(encode_ep1, 11880, "data/pattern_tables/ep1_pattern_db.npy", max_depth=20, phase2_restriction=True)       # C(12, 4)*4!

    print("Building EP2 Table...")
    build_pdb(encode_ep2, 40320, "data/pattern_tables/ep2_pattern_db.npy", max_depth=20, phase2_restriction=True)       # 

    print("All pattern databases built successfully.")

