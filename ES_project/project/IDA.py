# IDA*_parallel.py
# phase1(parallel) + phase2(parallel) 
from multiprocessing import Process, Queue
from .cube_state import rubik_cube
from .heuristic import heuristic_phase1, is_phase1_goal, heuristic_phase2, is_phase2_goal
from .heuristic import scramble_cube
from .IDAs import phase2_search_single
import copy
from queue import Empty
from time import time

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
def ida_phase2_worker(start_move, init_cube, max_depth, q):
    face, turn = start_move
    cube = copy.deepcopy(init_cube)
    cube.rotate(face, turn)
    path = [(face, turn)]

    def dfs(cube, g, threshold, last_face):
        h = heuristic_phase2(cube)
        f = g + h
        if f > threshold:
            return f, None
        if h == 0 and is_phase2_goal(cube):
            return f, path.copy()

        min_threshold = float('inf')
        for f_id in range(1, 6):
            if f_id == last_face or f_id == opposite_face[last_face]:
                continue
            turns = [1, 2, 3] if f_id in [0, 1] else [2]
            for t in turns:
                cube.rotate(f_id, t)
                path.append((f_id, t))
                f_next, result = dfs(cube, g + 1, threshold, f_id)
                if result is not None:
                    return f_next, result
                path.pop()
                cube.undo_rotate(f_id, t)
                min_threshold = min(min_threshold, f_next)
        return min_threshold, None

    threshold = heuristic_phase2(cube)
    print(f"[{face},{turn}] Start worker | Threshold={threshold}")

    while threshold <= max_depth:
        f_val, result = dfs(cube, 1, threshold, -1)
        if result is not None:
            print(f"[{face},{turn}] ✅ Found solution: {len(result)} moves at threshold {threshold}")
            q.put(result)
            return
        print(f"[{face},{turn}] Threshold {threshold} failed, next: {f_val}")
        threshold = f_val

    print(f"[{face},{turn}] ❌ No solution up to max_depth")
    q.put(None)
    # return

def phase2_search_parallel(cube, max_depth=17, timeout=100, fallback=True, num_solutions=3):
    q = Queue()
    processes = []

    face_turns = [(f, t) for f in range(1, 6) for t in ([1, 2, 3] if f in [0, 1] else [2])]
    for move in face_turns:
        p = Process(target=ida_phase2_worker, args=(move, cube, max_depth, q))
        p.start()
        processes.append(p)

    results = []
    start_time = time()

    try:
        while len(results) < num_solutions:
            result = q.get(timeout=timeout)
            if result:
                results.append(result)
                print(f"✅ Collected solution #{len(results)}: {len(result)} moves")
    except Empty:
        print("⏰ Timeout reached or not enough solutions found")

    for p in processes:
        p.terminate()

    if results:
        best = min(results, key=len)
        print(f"🎯 Best solution of {len(results)}: {len(best)} moves")
        return best
    elif fallback:
        print("↩️ Fallback to single-threaded Phase 2 search")
        return phase2_search_single(cube, max_depth=max_depth, verbose=True)
    else:
        return None

def solve_cube():
    cube = rubik_cube.cube_t()
    scramble_cube(cube)                         # L' F2 D R' F L D' B L D'

    print("======== Start Phase 1 Search... ========")
    phase1_solution = phase1_search_parallel(cube, max_depth=10)
    print("Phase 1:", phase1_solution)

    for face, turn in phase1_solution:
        cube.rotate(face, turn)
    print(cube.getBlocks_info())

    print("======== Start Phase 2 Search... ========")
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
