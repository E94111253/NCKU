from .cube_state import rubik_cube
from .solver import solve_cube

def build_info(scan_color):
    pick_cp = [
                [scan_color[0][8], scan_color[5][0], scan_color[2][2]],     # URF
                [scan_color[0][6], scan_color[2][0], scan_color[4][2]],     # UFL
                [scan_color[0][0], scan_color[4][0], scan_color[3][2]],     # ULB
                [scan_color[0][2], scan_color[3][0], scan_color[5][2]],     # UBR   
                [scan_color[1][2], scan_color[2][8], scan_color[5][6]],     # DFR
                [scan_color[1][0], scan_color[4][8], scan_color[2][6]],     # DLF
                [scan_color[1][6], scan_color[3][8], scan_color[4][6]],     # DBL
                [scan_color[1][8], scan_color[5][8], scan_color[3][6]]      # DRB
            ]
    pick_ep = [
                [scan_color[0][5], scan_color[5][1]],         # UR
                [scan_color[0][7], scan_color[2][1]],         # UF         
                [scan_color[0][3], scan_color[4][1]],         # UL
                [scan_color[0][1], scan_color[3][1]],         # UB
                [scan_color[1][5], scan_color[5][7]],         # DR
                [scan_color[1][1], scan_color[2][7]],         # DF
                [scan_color[1][3], scan_color[4][7]],         # DL
                [scan_color[1][7], scan_color[3][7]],         # DB
                [scan_color[2][5], scan_color[5][3]],         # FR
                [scan_color[2][3], scan_color[4][5]],         # FL
                [scan_color[3][5], scan_color[4][3]],         # BL
                [scan_color[3][3], scan_color[5][5]],         # BR
            ]

    center_color = [scan_color[i][4] for i in range(6)]

    final_cp = [
                [scan_color[0][4], scan_color[5][4], scan_color[2][4]],
                [scan_color[0][4], scan_color[2][4], scan_color[4][4]],
                [scan_color[0][4], scan_color[4][4], scan_color[3][4]],
                [scan_color[0][4], scan_color[3][4], scan_color[5][4]],
                [scan_color[1][4], scan_color[2][4], scan_color[5][4]],
                [scan_color[1][4], scan_color[4][4], scan_color[2][4]],
                [scan_color[1][4], scan_color[3][4], scan_color[4][4]],
                [scan_color[1][4], scan_color[5][4], scan_color[3][4]]
            ]
    
    final_ep = [
                [scan_color[0][4], scan_color[5][4]],         # UR
                [scan_color[0][4], scan_color[2][4]],         # UF         
                [scan_color[0][4], scan_color[4][4]],         # UL
                [scan_color[0][4], scan_color[3][4]],         # UB
                [scan_color[1][4], scan_color[5][4]],         # DR
                [scan_color[1][4], scan_color[2][4]],         # DF
                [scan_color[1][4], scan_color[4][4]],         # DL
                [scan_color[1][4], scan_color[3][4]],         # DB
                [scan_color[2][4], scan_color[5][4]],         # FR
                [scan_color[2][4], scan_color[4][4]],         # FL
                [scan_color[3][4], scan_color[4][4]],         # BL
                [scan_color[3][4], scan_color[5][4]],         # BR
            ]

    def color_to_corner(pick_cp, final_cp, center_color):
        cp = [0] * 8
        for i in range(8):
            j = 0
            for k in range(8):
                if pick_cp[i][j] in final_cp[k]:
                    if pick_cp[i][j+1] in final_cp[k]:
                        if pick_cp[i][j+2] in final_cp[k]:
                            cp[i] = k
        
        co = [0] * 8
        for i in range(8):
            if pick_cp[i][0] == center_color[0] or pick_cp[i][0] == center_color[1]:
                co[i] = 0
            elif pick_cp[i][1] == center_color[0] or pick_cp[i][1] == center_color[1]:
                co[i] = 1
            else :
                co[i] = 2
   
        return (cp, co)

    def color_to_edge(pick_ep, final_ep, center_color):
        ep = [0] * 12
        for i in range(12):
            j = 0
            for k in range(12):
                if pick_ep[i][j] in final_ep[k]:
                    if pick_ep[i][j+1] in final_ep[k]:
                        ep[i] = k

        eo = [0] * 12
        for i in range(12):
            if pick_ep[i][0] == center_color[0] or pick_ep[i][0] == center_color[1]:
                eo[i] = 0
            elif pick_ep[i][0] == center_color[2] or pick_ep[i][0] == center_color[3]:
                if pick_ep[i][1] == center_color[4] or pick_ep[i][1] == center_color[5]:
                    eo[i] = 0
                else:
                    eo[i] = 1   
            else:
                eo[i] = 1

        return (ep, eo)

    corner1 = color_to_corner(pick_cp, final_cp, center_color)
    edge1 = color_to_edge(pick_ep, final_ep, center_color)
    return corner1, edge1

if __name__ == "__main__":
    print("Test face color to cube state")
    colors = [
        ['Y', 'G', 'G', 'O', 'Y', 'W', 'G', 'G', 'O'],      # Face_u
        ['B', 'O', 'B', 'W', 'W', 'B', 'R', 'O', 'G'],      # Face_d
        ['W', 'W', 'B', 'Y', 'O', 'R', 'R', 'G', 'W'],      # Face_f
        ['O', 'R', 'G', 'W', 'R', 'Y', 'Y', 'B', 'W'],      # Face_b
        ['O', 'Y', 'R', 'R', 'G', 'G', 'B', 'R', 'Y'],      # Face_l
        ['Y', 'O', 'W', 'B', 'B', 'B', 'O', 'Y', 'R'],      # Face_r
    ]   
    final_co = [0] * 8
    final_eo = [0] * 12
    # print(pick_cp[0][2])
    # pick_cp, pick_ep, final_cp, final_ep, center_color = build_info(colors)
    #
    corner, edge = build_info(colors)
    cube = rubik_cube.cube_t()
    cube.cp = corner[0]   # 角塊位置
    cube.co = corner[1]   # 角塊方向
    cube.ep = edge[0]     # 邊塊位置
    cube.eo = edge[1]     # 邊塊方向 

    print(cube.getBlocks_info())

    # solve_cube(cube)