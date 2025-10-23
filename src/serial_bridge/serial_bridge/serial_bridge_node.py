# 파일 이름: serial_bridge_node.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from std_msgs.msg import String
import serial
import time

class SerialBridge(Node):
    def __init__(self):
        super().__init__('arduino_serial_bridge')
        
        # --- 파라미터 설정 ---
        # 아두이노가 연결된 시리얼 포트 (Linux: /dev/ttyUSB0, Windows: COM3 등)
        self.declare_parameter('port', '/dev/ttyACM0')
        # 통신 속도 (아두이노 코드와 반드시 일치해야 함)
        self.declare_parameter('baud_rate', 115200)
        
        port = self.get_parameter('port').get_parameter_value().string_value
        baud_rate = self.get_parameter('baud_rate').get_parameter_value().integer_value

        self.get_logger().info(f'시리얼 포트 {port} (Baud: {baud_rate})에 연결 시도 중...')

        try:
            # 시리얼 포트 열기
            self.serial_port = serial.Serial(port, baud_rate, timeout=1.0)
            time.sleep(2) # 아두이노가 리셋될 시간을 줌
            self.get_logger().info('시리얼 포트 연결 성공.')
        except serial.SerialException as e:
            self.get_logger().error(f'시리얼 포트 연결 실패: {e}')
            rclpy.shutdown()
            return

        # --- Subscriber 생성 ---
        self.subscription_j2 = self.create_subscription(
            Float64,
            '/joint_states/joint2',
            self.joint2_callback,
            10)
            
        self.subscription_j3 = self.create_subscription(
            Float64,
            '/joint_states/joint3',
            self.joint3_callback,
            10)
        
         # ✅ Gripper Command를 위한 Subscriber 추가
        self.subscription_gripper = self.create_subscription(
            String,
            '/gripper_command',
            self.gripper_callback,
            10)

        self.get_logger().info('Arduino Serial Bridge 노드가 시작되었습니다.')

    def joint2_callback(self, msg):
        angle = msg.data
        # 아두이노로 보낼 명령어 생성 (예: "J2:90.5\n")
        command = f"J2:{angle:.2f}\n" 
        self.serial_port.write(command.encode('utf-8')) # utf-8로 인코딩하여 전송
        self.get_logger().info(f'Sent to Arduino: {command.strip()}')

    def joint3_callback(self, msg):
        angle = msg.data
        command = f"J3:{angle:.2f}\n"
        self.serial_port.write(command.encode('utf-8'))
        self.get_logger().info(f'Sent to Arduino: {command.strip()}')
    
    def gripper_callback(self, msg):
        # 아두이노로 보낼 명령어 생성 (예: "G:open\n")
        command = f"G:{msg.data}\n"
        self.serial_port.write(command.encode('utf-8'))
        self.get_logger().info(f'Sent to Arduino: {command.strip()}')

    def destroy_node(self):
        # 노드 종료 시 시리얼 포트 닫기
        if hasattr(self, 'serial_port') and self.serial_port.is_open:
            self.serial_port.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    serial_bridge = SerialBridge()
    try:
        rclpy.spin(serial_bridge)
    except KeyboardInterrupt:
        pass
    finally:
        serial_bridge.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()