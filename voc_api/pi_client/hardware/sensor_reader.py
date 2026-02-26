"""
MQ6 + MEMS Odor Sensor Reader via ADS1115 ADC.

Hardware wiring:
    MQ6  analog out  → ADS1115 channel A0
    MEMS analog out  → ADS1115 channel A1
    ADS1115 VDD      → Pi 3.3V  (pin 1)
    ADS1115 GND      → Pi GND   (pin 6)
    ADS1115 SDA      → Pi GPIO2 (pin 3)
    ADS1115 SCL      → Pi GPIO3 (pin 5)

Enable I2C first:
    sudo raspi-config → Interface Options → I2C → Enable
    sudo reboot
    i2cdetect -y 1   # should show 0x48
"""

import time
import logging
import numpy as np

log = logging.getLogger(__name__)


class SensorReader:
    """
    Reads MQ6 and MEMS odor sensors via ADS1115 ADC over I2C.
    Falls back to simulated data if hardware is unavailable.
    """

    def __init__(self, simulate: bool = False):
        self.simulate = simulate
        self.mq6_channel = None
        self.mems_channel = None
        self._t0 = time.time()

        if not simulate:
            self._init_hardware()

    # ── hardware init ────────────────────────────────────────────
    def _init_hardware(self):
        """Try to initialize ADS1115. Falls back to simulate on failure."""
        try:
            import board
            import busio
            import adafruit_ads1x15.ads1115 as ADS
            from adafruit_ads1x15.analog_in import AnalogIn

            i2c = busio.I2C(board.SCL, board.SDA)
            ads = ADS.ADS1115(i2c)

            # MQ6 on channel 0, MEMS odor on channel 1
            self.mq6_channel  = AnalogIn(ads, ADS.P0)
            self.mems_channel = AnalogIn(ads, ADS.P1)

            log.info("✅ ADS1115 initialized  (MQ6 → A0, MEMS → A1)")

        except Exception as e:
            log.warning(f"⚠ Hardware init failed: {e}")
            log.warning("  Falling back to simulated sensor data.")
            self.simulate = True

    # ── read one sample ──────────────────────────────────────────
    def read_one(self) -> tuple[float, float]:
        """
        Read a single voltage from each sensor.
        Returns (mq6_voltage, mems_voltage).
        """
        if self.simulate:
            return self._simulate()

        mq6_v  = self.mq6_channel.voltage
        mems_v = self.mems_channel.voltage
        return float(mq6_v), float(mems_v)

    # ── simulation ───────────────────────────────────────────────
    def _simulate(self) -> tuple[float, float]:
        """
        Simulate realistic VOC sensor readings.
        Adds drift + noise to mimic real sensor behaviour.
        """
        t = time.time() - self._t0
        mq6_base  = 3.5  + 0.05 * np.sin(t / 20)
        mems_base = 17.5 + 0.10 * np.sin(t / 15)
        mq6_v  = mq6_base  + np.random.normal(0, 0.01)
        mems_v = mems_base + np.random.normal(0, 0.05)
        return float(mq6_v), float(mems_v)

    # ── collect a full window ────────────────────────────────────
    def collect_window(
        self,
        duration_sec: int = 30,
        rate_sec: float = 0.1,
        hand_check_fn=None,
        on_sample_fn=None,
    ) -> tuple[list[float], list[float]]:
        """
        Collect samples for `duration_sec` seconds at `rate_sec` intervals.

        Args:
            duration_sec:   Length of the sampling window.
            rate_sec:       Seconds between samples.
            hand_check_fn:  Optional callable → bool. If provided, pauses
                            sampling when it returns False (hand removed).
            on_sample_fn:   Optional callable(n_samples) for progress updates.

        Returns:
            (mq6_samples, mems_samples) — two lists of voltage readings.
        """
        mq6_samples: list[float]  = []
        mems_samples: list[float] = []

        t_start = time.time()

        while time.time() - t_start < duration_sec:
            # Pause if hand is removed
            if hand_check_fn and not hand_check_fn():
                log.info("⏸  Hand removed — pausing sampling")
                while hand_check_fn and not hand_check_fn():
                    time.sleep(0.2)
                log.info("▶  Hand re-detected — resuming")

            mq6_v, mems_v = self.read_one()
            mq6_samples.append(mq6_v)
            mems_samples.append(mems_v)

            if on_sample_fn:
                on_sample_fn(len(mq6_samples))

            time.sleep(rate_sec)

        return mq6_samples, mems_samples

    @property
    def is_simulated(self) -> bool:
        return self.simulate
