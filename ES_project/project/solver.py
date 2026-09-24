# solver.py

from .cube_state import rubik_cube
from .heuristic import *
from .search import phase1_search_parallel
from .IDA import phase2_search_parallel
from .IDAs import phase2_search_single
import time

def solve_cube(cube_state=None, scramble: bool = True, max_depth1=11, max_depth2=17):
    # 初始化魔術方塊
    cube = rubik_cube.cube_t()

    if cube_state is not None:
        cube = cube_state
        print("Initial Cube State (from input):")
        print(cube.getBlocks_info())
    # 打亂魔術方塊
    elif scramble:
        scramble_cube(cube)                 # L' F2 D R' F L D' B L D'
        print("Scrambled Cube:")
        print(cube.getBlocks_info())

    # ========== Phase 1 搜尋 ==========
    print("======== Start Phase 1 Search... ========")
    t1 = time.time()
    phase1_solution = phase1_search_parallel(cube, max_depth=max_depth1)        # 10
    t2 = time.time()

    if phase1_solution is None:
        print("Phase 1 failed to find solution.")
        return

    print(f"Phase 1 complete in {len(phase1_solution)} moves ({t2 - t1:.2f}s): {phase1_solution}")
    apply_moves(cube, phase1_solution)
    print("After Phase 1:")
    print(cube.getBlocks_info())

    # ========== Phase 2 搜尋 ==========
    print("======== Start Phase 2 Search (IDA*)... ========")
    t3 = time.time()
    phase2_solution = phase2_search_parallel(cube, max_depth=max_depth2)
    # phase2_solution = phase2_search_single(cube, max_depth=16, verbose=True)
    t4 = time.time()

    if phase2_solution is None:
        print("Phase 2 failed to find solution.")
        return

    print(f"Phase 2 solution ({len(phase2_solution)} moves ({t4 - t3:.2f}s): {phase2_solution}")
    apply_moves(cube, phase2_solution)
    print("Final Cube State:")
    print(cube.getBlocks_info())

    # ========== 完整解 ==========
    full_solution = phase1_solution + phase2_solution
    print(f"\nTotal Moves: {len(full_solution)}")
    print("Solution:", full_solution)
    print(f"Total time: {t4 - t1:.2f} seconds")

    return full_solution

if __name__ == "__main__":
    solve_cube()
