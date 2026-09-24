# phase1
from multiprocessing import Process, Queue
from .cube_state import rubik_cube
from .heuristic import heuristic_phase1, is_phase1_goal, heuristic_phase2, is_phase2_goal
from .heuristic import scramble_cube
import copy

opposite_face = [1, 0, 3, 2, 5, 4]

def dfs_phase1_worker(start_move, cube, max_depth, q):
    face, turn = start_move
    cube.rotate(face, turn)
    path = [(face, turn)]

    def dfs(cube, depth, last_face):
        h = heuristic_phase1(cube)
        if depth + h > max_depth:
            return None
        if h == 0 and is_phase1_goal(cube):
            return path.copy()

        for f in range(1, 6):
            if f == last_face or f == opposite_face[last_face]:
                continue
            for t in [1, 2, 3]:
                cube.rotate(f, t)
                path.append((f, t))
                result = dfs(cube, depth + 1, f)
                if result:
                    return result
                path.pop()
                cube.undo_rotate(f, t)
        return None

    result = dfs(cube, 1, face)
    if result:
        q.put(result)

def phase1_search_parallel(init_cube, max_depth):
    q = Queue()
    processes = []

    for face in range(1, 6):
        for turn in [1, 2, 3]:
            cube_copy = copy.deepcopy(init_cube)
            p = Process(target=dfs_phase1_worker, args=((face, turn), cube_copy, max_depth, q))
            p.start()
            processes.append(p)

    result = q.get()
    for p in processes:
        p.terminate()
    return result
# ===========================================================================
def dfs_phase2_worker(start_move, init_cube, max_depth, q):
    face, turn = start_move
    cube = copy.deepcopy(init_cube)
    cube.rotate(face, turn)
    path = [(face, turn)]

    def dfs(cube, depth, last_face):
        h = heuristic_phase2(cube)
        if depth + h > max_depth:
            return False
        if h == 0 and is_phase2_goal(cube):
            return True

        for f in range(1, 6):
            if f == last_face or f == opposite_face[last_face]:
                continue
            turns = [1, 2, 3] if f in [0, 1] else [2]
            for t in turns:
                cube.rotate(f, t)
                path.append((f, t))
                if dfs(cube, depth + 1, f):
                    return True
                path.pop()
                cube.undo_rotate(f, t)
        return False

    if dfs(cube, 0, face):
        q.put(path.copy())

def phase2_search_parallel(cube, max_depth=12):
    q = Queue()
    processes = []

    for face in range(1, 6):
        turns = [1, 2, 3] if face in [0, 1] else [2]
        for turn in turns:
            p = Process(target=dfs_phase2_worker, args=((face, turn), cube, max_depth, q))
            p.start()
            processes.append(p)

    from queue import Empty
    try:
        result = q.get(timeout=200)
        for p in processes:
            p.terminate()
        return result
    except Empty:
        print("❌ Timeout: No Phase 2 solution found.")
        for p in processes:
            p.terminate()
        return None
    

def solve_cube():
    cube = rubik_cube.cube_t()
    scramble_cube(cube)                         # L' F2 D R' F L D' B L D'

    print("Start Phase 1 search...")
    phase1_solution = phase1_search_parallel(cube, max_depth=10)
    print("Phase 1:", phase1_solution)

    for face, turn in phase1_solution:
        cube.rotate(face, turn)
    print(cube.getBlocks_info())

    print("Start Phase 2 search...")
    phase2_solution = phase2_search_parallel(cube, max_depth=17)

    if phase2_solution is None:
        print("Failed to solve Phase 2 within time limit.")
        return
    print("Phase 2:", phase2_solution)

    final_solution = phase1_solution + phase2_solution
    print("Final solution:", final_solution)

    for face, turn in phase2_solution:
        cube.rotate(face, turn)

    print("Final Cube State:")
    print(cube.getBlocks_info())

if __name__ == "__main__":
    from time import time
    t0 = time()
    solve_cube()
    print(f"🕒 Solved in {time() - t0:.2f} seconds")
