# cube state

class rubik_cube:
    def __init__(self):
        pass
    
    class face_type:
        top    = 0
        bottom = 1
        front  = 2
        back   = 3
        left   = 4
        right  = 5
    
    class cube_t:
        def __init__(self):         # 初始化            
            self.cp = [i for i in range(8)]
            self.co = [0] * 8
            self.ep = [i for i in range(12)]
            self.eo = [0] * 12
        
        def get_corner(self):
            return self.cp, self.co

        def get_edge(self):
            return self.ep, self.eo
            
        def rotate(self, face , count : int = 1):
            corner_rotate_map = [
                        [ 0, 1, 2, 3 ],     #top
                        [ 7, 6, 5, 4 ],     #bottom
                        [ 4, 5, 1, 0 ],     #front
                        [ 6, 7, 3, 2 ],     #back
                        [ 5, 6, 2, 1 ],     #left
                        [ 7, 4, 0, 3 ]      #right
            ]
                
            edge_rotate_map = [
                        [ 0,  1,  2,  3 ],  # top
                        [ 7,  6,  5,  4 ],  # bottom
                        [ 8,  5,  9,  1 ],  # front
                        [10,  7, 11,  3 ],  # back
                        [ 6, 10,  2,  9 ],  # left
                        [ 4,  8,  0, 11 ]   # right
            ]
            def swap(A,B):
                temp = A[B[3]]
                A[B[3]] = A[B[2]]
                A[B[2]] = A[B[1]]
                A[B[1]] = A[B[0]]
                A[B[0]] = temp  
                
                return A

            def swap180(a,b):
                return b, a
            # 正規化
            count = count % 4 
            corners = corner_rotate_map[face]   
            edges = edge_rotate_map[face]

            if count == 2:
                self.cp[corners[0]], self.cp[corners[2]] = swap180(self.cp[corners[0]], self.cp[corners[2]])
                self.cp[corners[1]], self.cp[corners[3]] = swap180(self.cp[corners[1]], self.cp[corners[3]])
                self.co[corners[0]], self.co[corners[2]] = swap180(self.co[corners[0]], self.co[corners[2]])
                self.co[corners[1]], self.co[corners[3]] = swap180(self.co[corners[1]], self.co[corners[3]])

                self.ep[edges[0]], self.ep[edges[2]] = swap180(self.ep[edges[0]], self.ep[edges[2]])
                self.ep[edges[1]], self.ep[edges[3]] = swap180(self.ep[edges[1]], self.ep[edges[3]])
                self.eo[edges[0]], self.eo[edges[2]] = swap180(self.eo[edges[0]], self.eo[edges[2]])
                self.eo[edges[1]], self.eo[edges[3]] = swap180(self.eo[edges[1]], self.eo[edges[3]])

            else:
                for _ in range(count):
                    # 角塊旋轉（使用 position-indexed）
                    self.cp = swap(self.cp, corners)
                    self.co = swap(self.co, corners)
                    self.ep = swap(self.ep, edges)
                    self.eo = swap(self.eo, edges)

                # === 處理角塊方向 ===
                if face in [2, 3, 4, 5]:  # 只有側面會改變 co
                    for i in [0, 2]:
                        self.co[corners[i]] = (self.co[corners[i]] - 1) % 3
                    for i in [1, 3]:
                        self.co[corners[i]] = (self.co[corners[i]] + 1) % 3

                if face in [2, 3]:  # F、B：該面所有邊塊都翻轉
                    for i in edges:
                        self.eo[i] ^= 1
                    

        def inverse_move(self, face, count):
            return face, (4 - count) % 4    
        
        def undo_rotate(self, face, count=1):
            inv_face, inv_count = self.inverse_move(face, count)
            self.rotate(inv_face, inv_count)

        def getBlocks_info(self):
            return (self.cp, self.co), (self.ep, self.eo)
"""
 * observing from the top face, the index of corners will be like this
 *     *-*-*-*             *-*-*-*
 *     |2| |3|             |6| |7|
 *     *-*-*-*             *-*-*-*
 *     | | | |             | | | |
 *     *-*-*-*             *-*-*-*
 *     |1| |0| <-(UFR)     |5| |4|  <-(DFR)
 *     *-*-*-*             *-*-*-*
 * the top face,        the bottom face
 *
 * observing from the top face, the index of edges will be like this
 *     *-*-*-*          *-*-*-*         *-*-*-*
 *     | |3| |          |A| |B|         | |7| |
 *     *-*-*-*          *-*-*-*         *-*-*-*
 *     |2| |0|          | | | |         |6| |4|
 *     *-*-*-*          *-*-*-*         *-*-*-*
 *     | |1| |          |9| |8|         | |5| |
 *     *-*-*-*          *-*-*-*         *-*-*-*
 * the top face,    the middle level,   the bottom face
 *
 * the priority of the key faces: UD > LR > FB
 *
 """

 # 測試程式碼
if __name__ == "__main__":
    cube = rubik_cube.cube_t()
    print("角塊位置:", cube.get_corner()[0])
    print("角塊旋轉狀態:", cube.get_corner()[1])
    print("邊塊位置:", cube.get_edge()[0])
    print("邊塊旋轉狀態:", cube.get_edge()[1])

    print("Test rotations:")
    print("01", cube.getBlocks_info())
    cube.rotate(4, 3)
    print("02", cube.getBlocks_info())
    cube.rotate(0, 1)
    print("03",  cube.getBlocks_info())
    cube.rotate(4, 3)
    print("04", cube.getBlocks_info())
    cube.rotate(0, 3)
    print("05", cube.getBlocks_info())
    cube.rotate(4, 3)
    print("06", cube.getBlocks_info())
    cube.rotate(0, 3)
    print("07", cube.getBlocks_info())
    cube.rotate(4, 3)
    print("08", cube.getBlocks_info())
    cube.rotate(0, 1)
    print("09", cube.getBlocks_info())
    cube.rotate(4, 1)
    print("10", cube.getBlocks_info())
    cube.rotate(0, 1)
    print("11", cube.getBlocks_info())
    cube.rotate(4, 2)
    print("12", cube.getBlocks_info())

   