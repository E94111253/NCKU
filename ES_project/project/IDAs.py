# IDA*_single.py
# phase1(parallel) + phase2(single) 
from .heuristic import heuristic_phase2, is_phase2_goal
from .cube_state import rubik_cube
from .Database import encode_cp, encode_ep1, encode_ep2

opposite_face = [1, 0, 3, 2, 5, 4]

def phase2_search_single(cube, max_depth=16, verbose=False):
    path = []
    visited = set()
    node_count = 0

    def dfs(cube, g, threshold, last_face):
        nonlocal node_count
        node_count += 1

        h = heuristic_phase2(cube)
        f = g + h
        if f > threshold:
            return f, None
        if h == 0 and is_phase2_goal(cube):
            return f, path.copy()
        
        state_key = ( 
            encode_cp(cube),
            encode_ep1(cube),
            encode_ep2(cube)
        )
        if (state_key, g) in visited:
            return float("inf"), None
        visited.add((state_key, g))


        min_threshold = float('inf')
        for face in range(1, 6):
            if face == last_face or face == opposite_face[last_face]:
                continue
            turns = [1, 2, 3] if face in [0, 1] else [2]
            for t in turns:
                cube.rotate(face, t)
                path.append((face, t))
                f_next, result = dfs(cube, g + 1, threshold, face)
                if result is not None:
                    return f_next, result
                path.pop()
                cube.undo_rotate(face, t)
                min_threshold = min(min_threshold, f_next)
        return min_threshold, None

    threshold = heuristic_phase2(cube)
    #=========== debug ===========
    cp_val = encode_cp(cube)
    ep1_val = encode_ep1(cube)
    ep2_val = encode_ep2(cube)
    print(f"Encoded CP = {cp_val}, EP1 = {ep1_val}, EP2 = {ep2_val}")

    while threshold <= max_depth:
        path.clear()
        visited.clear()
        node_count = 0
        if verbose:
            print(f"IDA* search at threshold: {threshold}")
        f_val, result = dfs(cube, 0, threshold, -1)
        if result is not None:
            return result
        if f_val == float('inf'):
            break
        threshold = f_val
    return None

