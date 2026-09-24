import RPi.GPIO as GPIO
import time

# 馬達對應 GPIO
motor_pins = [22, 17, 19, 27, 26]           #[]
"""
[ca]--[17]--[  ]        [ca]--[Fr]--[  ]
|              |        |              |
[26]  [22]  [27]        [Ri]  [Dw]  [Le]
|              |        |              |
[  ]--[19]--[  ]        [  ]--[Ba]--[  ]   
"""
solution = [(4, 2), (0, 1), (2, 2), (0, 2), (1, 1), (3, 1), (1, 1)]
# 每90度轉動時間（根據實測調整）
DEGREE_TIME = 0.111  # 秒（每 90 度）

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