#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ROS2 → Arduino 시리얼 브리지 (LED 제어: "roka" | "enemy" | "none")
#
# - 구독: /led_control (std_msgs/String)
# - 송신: "<cmd>\n" 을 시리얼로 전송  (cmd는 소문자화/트림)
#
# 안정화:
#   * 포트 자동 재연결 (reopen_interval_ms)
#   * DTR/RTS 끔(자동 리셋 최소화)
#   * exclusive=True (Linux 중복오픈 방지)
#   * debug_tx 옵션으로 송신 로그

import time
import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import serial
from serial import SerialException


class LedSerialBridge(Node):
    def __init__(self) -> None:
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

        # 주기적으로 포트 재오픈 시도
        self.create_timer(self.reopen_interval_ms / 1000.0, self.try_open_timer)

        # 최초 오픈 시도
        self.try_open_timer()

    # ===== ROS 콜백 =====
    def cb_led(self, msg: String) -> None:
        # 앞/뒤 공백 제거, 소문자화
        cmd = (msg.data or '').strip().lower()
        if cmd not in ('roka', 'enemy', 'none'):
            self.get_logger().warn(f"알 수 없는 LED 명령: '{msg.data}'. (허용: roka|enemy|none)")
            return
        self._send_line(cmd + '\n')

    # ===== Serial Open/Close =====
    def try_open_timer(self) -> None:
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
                    exclusive=True,   # Linux에서 포트 단일 점유
                )
                # 자동 리셋 최소화
                try:
                    self._ser.dtr = False
                    self._ser.rts = False
                except Exception:
                    pass
                # 버퍼 비우기
                try:
                    self._ser.reset_input_buffer()
                    self._ser.reset_output_buffer()
                except Exception:
                    pass

                self.get_logger().info(f'시리얼 오픈: {self.port} @ {self.baud}')

                # (선택) RX 로그 받고 싶으면 스레드 켜기
                if not self._rx_running:
                    self._rx_running = True
                    self._rx_thread = threading.Thread(target=self.rx_loop, daemon=True)
                    self._rx_thread.start()

            except Exception as e:
                self._ser = None
                self.get_logger().warn(f'포트 오픈 실패: {e}')

    def close_serial(self) -> None:
        with self._ser_lock:
            if self._ser is not None:
                try:
                    self._ser.close()
                except Exception:
                    pass
            self._ser = None

    # ===== TX/RX =====
    def _send_line(self, line: str) -> None:
        with self._ser_lock:
            ser = self._ser
        if ser is None or not getattr(ser, 'is_open', False):
            self.get_logger().warn('시리얼 미연결 상태—전송 건너뜀')
            return
        try:
            ser.write(line.encode('ascii'))
            if self.debug_tx:
                self.get_logger().info(f"TX: {line.strip()}")
        except Exception as e:
            self.get_logger().error(f'전송 실패: {e}')
            self.close_serial()

    def rx_loop(self) -> None:
        # 아두이노가 print한 디버그 라인을 보고 싶을 때 유용
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
                    self.get_logger().warn(f'RX 에러: {e}')
                self.close_serial()
                time.sleep(0.1)

    # ===== 종료 =====
    def shutdown(self) -> None:
        self._shutting_down = True
        self._rx_running = False
        self.close_serial()
        time.sleep(0.05)


def main() -> None:
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
