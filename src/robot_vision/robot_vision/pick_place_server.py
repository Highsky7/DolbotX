#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from mtc_interfaces.srv import PickPlace # PickPlace 서비스 타입을 임포트합니다.

class PickPlaceServerNode(Node):
    def __init__(self):
        super().__init__('pick_place_server_node')
        # 'pick_place' 라는 이름으로 서비스를 생성합니다.
        # 클라이언트가 요청하면 self.pick_place_callback 함수가 실행됩니다.
        self.srv = self.create_service(
            PickPlace, 
            'pick_place_service', 
            self.pick_place_callback)
        self.get_logger().info('✅ Pick and Place service server is ready.')

    def pick_place_callback(self, request, response):
        # 클라이언트로부터 받은 요청 데이터를 로그로 출력합니다.
        self.get_logger().info(
            f'Incoming request received: \n'
            f'  x: {request.x:.3f}\n'
            f'  y: {request.y:.3f}\n'
            f'  z: {request.z:.3f}'
        )
        
        # 여기에 실제로 로봇팔을 움직이거나 하는 로직을 추가할 수 있습니다.
        # 지금은 성공적으로 받았다는 응답만 보내줍니다.
        response.success = True
        response.message = "Successfully received the coordinates."
        
        # 처리 결과를 클라이언트에게 반환합니다.
        return response

def main(args=None):
    rclpy.init(args=args)
    node = PickPlaceServerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()