"""
Fan controller for flushing the VOC sampling chamber.

Hardware wiring (typical):
    Fan control → GPIO 27 (pin 13) via MOSFET/relay  [configurable]
    Fan power   → External 5V/12V supply
    MOSFET gate → GPIO 27
"""

import logging
import time
import threading

log = logging.getLogger(__name__)

DEFAULT_PIN = 27


class FanManager:
    """
    Controls a fan to flush the sensor chamber between readings.
    Falls back to a time-delay stub if GPIO is unavailable.
    """

    def __init__(self, pin: int = DEFAULT_PIN, simulate: bool = False):
        self.pin = pin
        self.simulate = simulate
        self._gpio = None
        self._running = False

        if not simulate:
            self._init_gpio()

    def _init_gpio(self):
        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.pin, GPIO.OUT)
            GPIO.output(self.pin, GPIO.LOW)   # fan OFF initially
            self._gpio = GPIO
            log.info(f"✅ Fan controller ready on GPIO {self.pin}")
        except Exception as e:
            log.warning(f"⚠ Fan GPIO init failed: {e}  — fan simulation active")
            self.simulate = True

    def start(self):
        """Turn the fan ON."""
        self._running = True
        if self._gpio:
            self._gpio.output(self.pin, self._gpio.HIGH)
        log.info("🌀 Fan ON")

    def stop(self):
        """Turn the fan OFF."""
        self._running = False
        if self._gpio:
            self._gpio.output(self.pin, self._gpio.LOW)
        log.info("⏹  Fan OFF")

    def flush(self, duration_sec: int = 20):
        """
        Run the fan for `duration_sec` seconds to clear residual VOC.
        Blocks the calling thread.
        """
        log.info(f"🌀 Flushing chamber for {duration_sec}s…")
        self.start()
        time.sleep(duration_sec)
        self.stop()
        log.info("✅ Chamber flush complete")

    def flush_async(self, duration_sec: int = 20):
        """Non-blocking flush in a background thread."""
        t = threading.Thread(target=self.flush, args=(duration_sec,), daemon=True)
        t.start()
        return t

    def cleanup(self):
        """Release GPIO resources."""
        self.stop()
        if self._gpio:
            try:
                self._gpio.cleanup(self.pin)
            except Exception:
                pass

    @property
    def is_running(self) -> bool:
        return self._running
