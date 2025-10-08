#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
ROS2 to Arduino Serial Bridge for LED Control.

This node subscribes to a topic to receive LED control commands and sends them
over a serial connection to an Arduino or similar microcontroller.

- Subscribes to: /led_control (std_msgs/String)
- Transmits: "<cmd>\n" over serial, where cmd is a lowercase, trimmed string.
  (e.g., "roka", "enemy", "none")

Stabilization Features:
  * Automatic port reconnection (reopen_interval_ms).
  * DTR/RTS disabled to minimize automatic resets on Arduinos.
  * `exclusive=True` to prevent multiple processes from opening the port on Linux.
  * `debug_tx` option for logging transmitted data.
"""

import time
import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import serial
from serial import SerialException


class LedSerialBridge(Node):
    """
    Act as a bridge between a ROS topic and a serial port for LED control.
    """

    def __init__(self) -> None:
        """
        Initialize the LedSerialBridge node.

        Declares parameters, sets up the ROS subscriber, and starts a timer
        to periodically attempt to open the serial port.
        """
        super().__init__('led_serial_bridge')

        # ---------- Parameters ----------
        self.declare_parameter('port', '/dev/LED')
        self.declare_parameter('baud', 115200)
        self.declare_parameter('reopen_interval_ms', 1000)
        self.declare_parameter('debug_tx', False)
        self.declare_parameter('topic', '/led_control')

        self.port = str(self.get_parameter('port').value)
        self.baud = int(self.get_parameter('baud').value)
        self.reopen_interval_ms = int(self.get_parameter('reopen_interval_ms').value)
        self.debug_tx = bool(self.get_parameter('debug_tx').value)
        self.topic = str(self.get_parameter('topic').value)

        # ---------- State ----------
        self._ser = None
        self._ser_lock = threading.Lock()
        self._rx_running = False
        self._rx_thread = None
        self._shutting_down = False

        # ---------- ROS I/O ----------
        self.create_subscription(String, self.topic, self.cb_led, 10)

        # Periodically attempt to reopen the port
        self.create_timer(self.reopen_interval_ms / 1000.0, self.try_open_timer)

        # Initial open attempt
        self.try_open_timer()

    def cb_led(self, msg: String) -> None:
        """
        Receive a command from the /led_control topic and send it via serial.

        Args:
            msg (String): The message containing the command ('roka', 'enemy', or 'none').
        """
        # Trim whitespace and convert to lowercase
        cmd = (msg.data or '').strip().lower()
        if cmd not in ('roka', 'enemy', 'none'):
            self.get_logger().warn(f"Unknown LED command: '{msg.data}'. (Allowed: roka|enemy|none)")
            return
        self._send_line(cmd + '\n')

    def try_open_timer(self) -> None:
        """
        Try to open the serial port periodically.

        This allows the node to automatically reconnect if the device is unplugged
        and plugged back in.
        """
        if self._shutting_down:
            return
        with self._ser_lock:
            if self._ser is not None and getattr(self._ser, 'is_open', False):
                return
            try:
                self._ser = serial.Serial(
                    self.port,
                    self.baud,
                    timeout=0.05,
                    write_timeout=0.2,
                    exclusive=True,   # Ensure single access to the port on Linux
                )
                # Minimize automatic resets on Arduino
                try:
                    self._ser.dtr = False
                    self._ser.rts = False
                except Exception:
                    pass
                # Clear buffers
                try:
                    self._ser.reset_input_buffer()
                    self._ser.reset_output_buffer()
                except Exception:
                    pass

                self.get_logger().info(f'Serial port opened: {self.port} @ {self.baud}')

                # Optional: Start a thread to read and log incoming serial data
                if not self._rx_running:
                    self._rx_running = True
                    self._rx_thread = threading.Thread(target=self.rx_loop, daemon=True)
                    self._rx_thread.start()

            except Exception as e:
                self._ser = None
                self.get_logger().warn(f'Failed to open port: {e}')

    def close_serial(self) -> None:
        """Safely close the serial port."""
        with self._ser_lock:
            if self._ser is not None:
                try:
                    self._ser.close()
                except Exception:
                    pass
            self._ser = None

    def _send_line(self, line: str) -> None:
        """
        Send a string over the serial port.

        Args:
            line (str): The string to send, which should end with a newline.
        """
        with self._ser_lock:
            ser = self._ser
        if ser is None or not getattr(ser, 'is_open', False):
            self.get_logger().warn('Serial not connected—skipping send')
            return
        try:
            ser.write(line.encode('ascii'))
            if self.debug_tx:
                self.get_logger().info(f"TX: {line.strip()}")
        except Exception as e:
            self.get_logger().error(f'Send failed: {e}')
            self.close_serial()

    def rx_loop(self) -> None:
        """
        Read and log data from the serial port in a loop.

        This is useful for viewing debug prints from the Arduino.
        """
        buf = bytearray()
        while self._rx_running and not self._shutting_down:
            with self._ser_lock:
                ser = self._ser
            if ser is None or not getattr(ser, 'is_open', False):
                time.sleep(0.05)
                continue
            try:
                chunk = ser.read(64)
                if not chunk:
                    continue
                buf.extend(chunk)
                while True:
                    idx = buf.find(b'\n')
                    if idx < 0:
                        break
                    line = buf[:idx].decode('ascii', errors='ignore').strip()
                    del buf[:idx + 1]
                    if line:
                        self.get_logger().info(f'RX: {line}')
            except (SerialException, OSError, ValueError) as e:
                if not self._shutting_down:
                    self.get_logger().warn(f'RX error: {e}')
                self.close_serial()
                time.sleep(0.1)

    def shutdown(self) -> None:
        """Clean up resources on node shutdown."""
        self._shutting_down = True
        self._rx_running = False
        self.close_serial()
        time.sleep(0.05)


def main() -> None:
    """Run the ROS2 node."""
    rclpy.init()
    node = LedSerialBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.shutdown()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()