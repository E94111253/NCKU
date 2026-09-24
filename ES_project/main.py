from project.cube_state import rubik_cube
from project.solver import solve_cube
from project.build import build_info
from vision.camara import take_photo
from vision.scan_face import *
import RPi.GPIO as GPIO
import time

def wait_for_enter(prompt="按 ENTER 鍵繼續..."):
    """等待使用者按下 ENTER 鍵"""
    input(prompt)

def test(mode):
    face_set = ['u', 'd', 'f', 'b', 'l', 'r']
    if mode == '1':
        print("Mode 1")
        colors = [
        ['Y', 'G', 'G', 'O', 'Y', 'W', 'G', 'G', 'O'],      # Face_u
        ['B', 'O', 'B', 'W', 'W', 'B', 'R', 'O', 'G'],      # Face_d
        ['W', 'W', 'B', 'Y', 'O', 'R', 'R', 'G', 'W'],      # Face_f
        ['O', 'R', 'G', 'W', 'R', 'Y', 'Y', 'B', 'W'],      # Face_b
        ['O', 'Y', 'R', 'R', 'G', 'G', 'B', 'R', 'Y'],      # Face_l
        ['Y', 'O', 'W', 'B', 'B', 'B', 'O', 'Y', 'R'],      # Face_r
         ]   
    
    elif mode == '2':
    # Step1, Take photo(save as "photo_{face}.jpg")
        for i in face_set :
            take_photo(i)

    # Step2, Scan color(save in 'colors)
        
        colors = make_list(face_set)                 # Check color.py h、s、v if need
    set = ['u', 'd', 'f', 'b', 'l', 'r']
    for f in range(6):
        print(f"Face {set[f]}: {colors[f]}")

    print("==== Color Matries ====")
    for i in range(6):
        print(colors[i])

    #Step3, Build full cube
    cube = rubik_cube.cube_t()
    corner, edge = build_info(colors)

    cube.cp = corner[0]   # 角塊位置
    cube.co = corner[1]   # 角塊方向
    cube.ep = edge[0]     # 邊塊位置
    cube.eo = edge[1]     # 邊塊方向 

    print(cube.getBlocks_info())

    #Step4, Solver
    solution = solve_cube(cube)
    wait_for_enter()
    #Step5, Change solve to servos
    motor_pins = [17, 27, 22, 19, 26]           #[]
    DEGREE_TIME = 0.11                          # 90 dergee   

    for i in range(len(solution)):
        trun_face = solution[i][0]
        turn_degree = DEGREE_TIME * solution[i][1]
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(motor_pins[trun_face-1], GPIO.OUT)

        pwm = GPIO.PWM(motor_pins[trun_face-1], 50)
        pwm.start(0)

        def set_time(rotate_time):
            duty_cycle = 5.0
            pwm.ChangeDutyCycle(duty_cycle)
            time.sleep(rotate_time)
            pwm.ChangeDutyCycle(0)

        set_time(turn_degree)
        time.sleep(1.0)

        pwm.stop()
        GPIO.cleanup()       

    print("All finish!")

if __name__ == '__main__':
    test(mode='1')
    
