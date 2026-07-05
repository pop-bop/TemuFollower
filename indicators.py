import RPi.GPIO as GPIO

from config import RED_LED_PIN, GREEN_LED_PIN, BUZZER_PIN


TONES = {
    "state_follow":      [(880, 40, 0)],
    "state_approach":    [(600, 70, 40), (600, 70, 0)],
    "state_spin_search": [(300, 120, 60), (300, 120, 60), (300, 120, 0)],
    "sharp_turn":        [(1000, 35, 0)],
    "marker_green":      [(784, 90, 40), (988, 120, 0)],
    "marker_red":        [(500, 180, 80), (320, 260, 0)],
    "manual_test":       [(660, 100, 0)],
}


def setup_indicators():
    GPIO.setup(RED_LED_PIN, GPIO.OUT)
    GPIO.setup(GREEN_LED_PIN, GPIO.OUT)
    GPIO.setup(BUZZER_PIN, GPIO.OUT)
    GPIO.output(RED_LED_PIN, GPIO.LOW)
    GPIO.output(GREEN_LED_PIN, GPIO.LOW)
    buzzer_pwm = GPIO.PWM(BUZZER_PIN, 440)
    buzzer_pwm.start(0)
    return buzzer_pwm


def new_led_state():
    return {"active": False, "on": False, "next_toggle": 0.0,
            "interval": 0.1, "blinks_left": 0, "hold_on": False}


def start_blink(led_state, blinks, interval_s, hold_on=False):
    led_state.update(active=True, on=False, next_toggle=0.0,
                      interval=interval_s, blinks_left=blinks, hold_on=hold_on)


def update_led(led_state, pin, now):
    if not led_state["active"] or now < led_state["next_toggle"]:
        return
    led_state["on"] = not led_state["on"]
    GPIO.output(pin, GPIO.HIGH if led_state["on"] else GPIO.LOW)
    led_state["next_toggle"] = now + led_state["interval"]
    if led_state["blinks_left"] is not None:
        led_state["blinks_left"] -= 1
        if led_state["blinks_left"] <= 0:
            led_state["active"] = False
            GPIO.output(pin, GPIO.HIGH if led_state["hold_on"] else GPIO.LOW)


def new_buzzer_state():
    return {"queue": [], "seg_start": None}


def play_tone(buzzer_state, name):
    buzzer_state["queue"] = list(TONES[name])
    buzzer_state["seg_start"] = None


def update_buzzer(buzzer_state, buzzer_pwm, now):
    if not buzzer_state["queue"]:
        return
    freq, on_ms, off_ms = buzzer_state["queue"][0]
    if buzzer_state["seg_start"] is None:
        buzzer_state["seg_start"] = now
        buzzer_pwm.ChangeFrequency(freq)
        buzzer_pwm.ChangeDutyCycle(50)
    elapsed_ms = (now - buzzer_state["seg_start"]) * 1000.0
    if elapsed_ms < on_ms:
        return
    if elapsed_ms < on_ms + off_ms:
        buzzer_pwm.ChangeDutyCycle(0)
        return
    buzzer_state["queue"].pop(0)
    buzzer_state["seg_start"] = None
    buzzer_pwm.ChangeDutyCycle(0)
