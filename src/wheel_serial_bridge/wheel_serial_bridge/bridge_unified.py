#!/usr/bin/env python3
# ROS2 ↔ Arduino 직렬 브리지 (VL/FB ASCII 프로토콜)
# - TX:  /cmd_vel_left, /cmd_vel_right  →  "VL <L[m/s]> <R[m/s]>\n"
# - RX:  "FB <rpmL> <rpmR>\n"           →  /wheel_rpm_left, /wheel_rpm_right
# 안정화:
#   * exclusive=True로 포트 단일 점유
#   * DTR/RTS False로 자동리셋 최소화
#   * 포트 끊김 자동 재오픈
#   * debug_tx 파라미터로 송신 라인 로그

import time
import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

import serial
from serial import SerialException


class BridgeUnified(Node):
    def __init__(self) -> None:
        super().__init__('wheel_serial_bridge_unified')

        # ---- Parameters ----
        self.declare_parameter('port', '/dev/ttyACM0')
        self.declare_parameter('baud', 57600)
        self.declare_parameter('tx_rate_hz', 50.0)
        self.declare_parameter('idle_timeout_ms', 200)
        self.declare_parameter('reopen_interval_ms', 1000)
        self.declare_parameter('debug_tx', False)

        self.port = str(self.get_parameter('port').value)
        self.baud = int(self.get_parameter('baud').value)
        self.tx_dt = 1.0 / float(self.get_parameter('tx_rate_hz').value)
        self.idle_timeout_ms = int(self.get_parameter('idle_timeout_ms').value)
        self.reopen_interval_ms = int(self.get_parameter('reopen_interval_ms').value)
        self.debug_tx = bool(self.get_parameter('debug_tx').value)

        # ---- State ----
        self._ser = None
        self._ser_lock = threading.Lock()
        self._rx_thread = None
        self._rx_running = False
        self._shutting_down = False

        self.left = 0.0
        self.right = 0.0
        now = self.now_ms()
        self.last_left_ms = now
        self.last_right_ms = now

        # ---- ROS I/O ----
        self.create_subscription(Float32, 'cmd_vel_left', self.cb_left, 10)
        self.create_subscription(Float32, 'cmd_vel_right', self.cb_right, 10)

        self.pub_rpm_l = self.create_publisher(Float32, 'wheel_rpm_left', 10)
        self.pub_rpm_r = self.create_publisher(Float32, 'wheel_rpm_right', 10)

        # ---- Timers ----
        # 주기 송신
        self.create_timer(self.tx_dt, self.tx_timer_cb)
        # 주기 재오픈 시도
        self.create_timer(self.reopen_interval_ms / 1000.0, self.try_open_timer)

        # 최초 오픈 시도
        self.try_open_timer()

    # ----------------- Callbacks -----------------
    def cb_left(self, msg: Float32) -> None:
        self.left = float(msg.data)
        self.last_left_ms = self.now_ms()

    def cb_right(self, msg: Float32) -> None:
        self.right = float(msg.data)
        self.last_right_ms = self.now_ms()

    # ----------------- Serial Open/Close -----------------
    def try_open_timer(self) -> None:
        if self._shutting_down:
            return
        with self._ser_lock:
            if self._ser is not None and getattr(self._ser, 'is_open', False):
                return
            try:
                # 포트 단일 점유 + 짧은 타임아웃
                self._ser = serial.Serial(
                    self.port,
                    self.baud,
                    timeout=0.05,
                    write_timeout=0.2,
                    exclusive=True  # Linux: 중복 접근 방지
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

                self.get_logger().info(f'Opened serial {self.port} @ {self.baud}')

                # RX 스레드 시작
                if not self._rx_running:
                    self._rx_running = True
                    self._rx_thread = threading.Thread(target=self.rx_loop, daemon=True)
                    self._rx_thread.start()

            except Exception as e:
                self._ser = None
                # launch 출력이 너무 시끄럽지 않도록 warn
                self.get_logger().warn(f'Open failed: {e}')

    def close_serial(self) -> None:
        with self._ser_lock:
            if self._ser is not None:
                try:
                    self._ser.close()
                except Exception:
                    pass
            self._ser = None

    # ----------------- TX/RX -----------------
    def tx_timer_cb(self) -> None:
        # 시간 초과 시 안전 정지
        now = self.now_ms()
        l = self.left
        r = self.right
        if (now - self.last_left_ms > self.idle_timeout_ms) or (now - self.last_right_ms > self.idle_timeout_ms):
            l = 0.0
            r = 0.0

        line = f'VL {l:.3f} {r:.3f}\n'

        with self._ser_lock:
            ser = self._ser
        if ser is None or not getattr(ser, 'is_open', False):
            return

        try:
            ser.write(line.encode('ascii'))
            if self.debug_tx:
                self.get_logger().info(f'TX: {line.strip()}')
        except Exception as e:
            self.get_logger().error(f'Write failed: {e}')
            self.close_serial()

    def rx_loop(self) -> None:
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
                    self.handle_line(line)
            except (SerialException, OSError, ValueError) as e:
                if not self._shutting_down:
                    self.get_logger().warn(f'RX error: {e}')
                self.close_serial()
                time.sleep(0.1)

    def handle_line(self, line: str) -> None:
        # 기대 포맷: "FB <rpmL> <rpmR>"
        if not line:
            return
        if line.startswith('FB'):
            parts = line.split()
            if len(parts) >= 3:
                try:
                    rpm_l = float(parts[1])
                    rpm_r = float(parts[2])
                    self.pub_rpm_l.publish(Float32(data=rpm_l))
                    self.pub_rpm_r.publish(Float32(data=rpm_r))
                except ValueError:
                    # 무시
                    pass

    # ----------------- Utils/Shutdown -----------------
    @staticmethod
    def now_ms() -> int:
        return int(time.time() * 1000)

    def shutdown(self) -> None:
        # 안전 종료: RX 루프 정지 후 포트 닫기
        self._shutting_down = True
        self._rx_running = False
        self.close_serial()
        # 잠깐 대기(스레드 정리)
        time.sleep(0.05)


def main() -> None:
    rclpy.init()
    node = BridgeUnified()
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
            # 이미 내려간 컨텍스트면 무시
            pass


if __name__ == '__main__':
    main()

