"""
Hand-presence detector via IR proximity sensor on GPIO.

Hardware wiring (typical):
    IR sensor OUT → GPIO 17 (pin 11)   [configurable]
    IR sensor VCC → 3.3V (pin 1)
    IR sensor GND → GND  (pin 6)

The IR sensor pulls the GPIO LOW when a hand is detected.
Adjust ACTIVE_LOW based on your specific sensor module.
"""

import logging
import time

log = logging.getLogger(__name__)

# Default GPIO pin for the hand-detection IR sensor
DEFAULT_PIN = 17
ACTIVE_LOW  = True   # True = sensor pulls LOW when hand is present


class HandController:
    """
    Detects hand presence using an IR proximity sensor.
    Falls back to 'always present' if GPIO is unavailable.
    """

    def __init__(self, pin: int = DEFAULT_PIN, simulate: bool = False):
        self.pin = pin
        self.simulate = simulate
        self._gpio = None

        if not simulate:
            self._init_gpio()

    def _init_gpio(self):
        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            self._gpio = GPIO
            log.info(f"✅ Hand sensor ready on GPIO {self.pin}")
        except Exception as e:
            log.warning(f"⚠ GPIO init failed: {e}  — hand detection disabled")
            self.simulate = True

    def hand_present(self) -> bool:
        """Returns True when a hand is detected above the sensor."""
        if self.simulate:
            return True   # always present in simulation

        state = self._gpio.input(self.pin)
        return (state == 0) if ACTIVE_LOW else (state == 1)

    def wait_for_hand(self, poll_interval: float = 0.2) -> None:
        """Block until a hand is detected."""
        if self.hand_present():
            return
        log.info("🖐  Waiting for hand…")
        while not self.hand_present():
            time.sleep(poll_interval)
        log.info("✅ Hand detected")

    def cleanup(self):
        """Release GPIO resources."""
        if self._gpio:
            try:
                self._gpio.cleanup(self.pin)
            except Exception:
                pass
