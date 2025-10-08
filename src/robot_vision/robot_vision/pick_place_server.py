#!/usr/bin/env python3
"""
ROS2 Service Server for a Pick and Place Task.

This script creates a simple ROS2 service server that provides a 'PickPlace'
service. The server waits for a client to send 3D coordinates (x, y, z),
logs the received coordinates, and returns a success response.

This server acts as a placeholder or a simple mock for a more complex robot
arm control system. It demonstrates the server-side implementation of a
custom service interface.
"""
import rclpy
from rclpy.node import Node
from mtc_interfaces.srv import PickPlace  # Import the custom service type.


class PickPlaceServerNode(Node):
    """
    A ROS2 node that provides a server for the PickPlace service.
    """

    def __init__(self):
        """
        Initialize the PickPlaceServerNode.

        This creates the service named 'pick_place_service' and waits for
        requests from clients.
        """
        super().__init__('pick_place_server_node')
        # Create a service with the name 'pick_place_service'.
        # The self.pick_place_callback function will be executed upon a request.
        self.srv = self.create_service(
            PickPlace,
            'pick_place_service',
            self.pick_place_callback)
        self.get_logger().info('✅ Pick and Place service server is ready.')

    def pick_place_callback(self, request, response):
        """
        Handle an incoming request to the PickPlace service.

        This function is called whenever a client sends a request. It logs the
        received coordinates and sends back a response indicating success.

        Args:
            request (PickPlace.Request): The request message from the client,
                                         containing x, y, and z coordinates.
            response (PickPlace.Response): The response message to be sent
                                           back to the client.

        Returns:
            PickPlace.Response: The populated response object.
        """
        # Log the data received from the client.
        self.get_logger().info(
            f'Incoming request received: \n'
            f'  x: {request.x:.3f}\n'
            f'  y: {request.y:.3f}\n'
            f'  z: {request.z:.3f}'
        )

        # Actual robot arm movement logic would be added here.
        # For now, it just sends a successful response.
        response.success = True
        response.message = "Successfully received the coordinates."

        # Return the result to the client.
        return response


def main(args=None):
    """The main entry point for the ROS2 node."""
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