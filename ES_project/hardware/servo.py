import RPi.GPIO as GPIO
import time

"""
[ca]--[17]--[  ]        [ca]--[Fr]--[  ]
|              |        |              |
[26]  [22]  [27]        [Ri]  [Dw]  [Le]
|              |        |              |
[  ]--[19]--[  ]        [  ]--[Ba]--[  ]   
"""

def mode(n):
    if n == 1:
        set = [22, 17, 19, 27, 26]

        for i in range(5):
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(set[i], GPIO.OUT)

            pwm = GPIO.PWM(set[i], 50)
            pwm.start(0)

            def set_time(rotate_time):
                duty_cycle = 5.0
                pwm.ChangeDutyCycle(duty_cycle)
                time.sleep(rotate_time)
                pwm.ChangeDutyCycle(0)

            set_time(0.11*2)
            time.sleep(1)

            pwm.stop()
            GPIO.cleanup()
    elif n == 2:
        pin = 26
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(pin, GPIO.OUT)

        pwm = GPIO.PWM(pin, 50)
        pwm.start(0)

        def set_time(rotate_time):
            duty_cycle = 5.0
            pwm.ChangeDutyCycle(duty_cycle)
            time.sleep(rotate_time)
            pwm.ChangeDutyCycle(0)

        set_time(0.112)
        time.sleep(1.5)

        pwm.stop()
        GPIO.cleanup()

mode(2)
