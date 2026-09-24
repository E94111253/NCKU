# heuristic
#
from .Database import *
import numpy as np
from .cube_state import rubik_cube

# ============ 載入 pattern databases ============
co_table = np.load("data/pattern_tables/pdb_co.npy")
eo_table = np.load("data/pattern_tables/pdb_eo.npy")
cp_table = np.load("data/pattern_tables/phase2_cp_table.npy")
ep1_table = np.load("data/pattern_tables/ep1_pattern_db.npy")
ep2_table = np.load("data/pattern_tables/ep2_pattern_db.npy")
slice_table = np.load("data/pattern_tables/pdb_slice.npy")

opposite_face = [1, 0, 3, 2, 5, 4]

#============ Phase 1 ============
def heuristic_phase1(cube):
    return max(
        co_table[encode_corner_orientation(cube)],
        eo_table[encode_edge_orientation(cube)],
        slice_table[encode_slice(cube)]
    )

def is_phase1_goal(cube):
    _, co = cube.get_corner()
    _, eo = cube.get_edge()
    return all(c == 0 for c in co) and all(e == 0 for e in eo)

#============ Phase 2 ============
def heuristic_phase2(cube):
    cp = cp_table[encode_cp(cube)]
    ep1_idx = encode_ep1(cube)
    ep2_idx = encode_ep2(cube)

    if ep1_idx == -1 or ep2_idx == -1:
        return float('inf')  

    ep1 = ep1_table[ep1_idx]
    ep2 = ep2_table[ep2_idx]
    # return int(np.ceil((cp + ep1 + ep2) / 2))   # 或 max(cp, ep1, ep2)
    return max(cp, ep1, ep2)

def is_phase2_goal(cube):
    cp, _ = cube.get_corner()
    ep, _ = cube.get_edge()
    return cp == list(range(8)) and ep == list(range(12))

# ============ Phase 1 到 Phase 2 的轉換 ============
def apply_moves(cube, moves):
    for face, turns in moves:
        cube.rotate(face, turns)

#============ set scramble cube ============
def scramble_cube(cube):
    moves = [(4, 3), (2, 2), (1, 1), (2, 3), (5, 2), (3, 1),
             (1, 3), (4, 2), (2, 1), (5, 3)]                    # L' F2 D F' R2 B D' R2 B D'    
    moves2 = [(4, 3), (2, 2), (1, 1), (5, 3), (2, 1), (4, 1),
             (1, 3), (3, 1), (4, 1), (1, 3)]                    # L' F2 D R' F L D' B L D'  
    for face, t in moves:
        cube.rotate(face, t)
    print(cube.getBlocks_info())

#============ Main ============ 
if __name__ == "__main__":
    cube = rubik_cube.cube_t()
    
    print(cube.getBlocks_info())
    scramble_cube(cube)
    
    