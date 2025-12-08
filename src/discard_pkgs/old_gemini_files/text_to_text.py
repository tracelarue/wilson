import asyncio
import os
import json
from google import genai
from google.genai import types
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
import math

# Remove the transforms3d import and use math directly
# from transforms3d.euler import euler2quat

GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
client = genai.Client(api_key=GOOGLE_API_KEY)
model = "gemini-2.0-flash-live-001"

# Define your tools - adding navigate_to_location
navigate_to_location = {
    "name": "navigate_to_location",
    "description": "Navigate the robot to a specific location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The location name to navigate to (e.g., 'kitchen', 'living room', 'bedroom')"
            },
            "x": {
                "type": "number",
                "description": "The x-coordinate on the map"
            },
            "y": {
                "type": "number",
                "description": "The y-coordinate on the map"
            }
        },
        "required": ["location", "x", "y"]
    },
}


# Add the new navigation tool to the tools list
tools = [{"function_declarations": [navigate_to_location]}]

# Configuring the AI with navigation information
config = {
    "system_instruction": types.Content(
        parts=[
            types.Part(
                text=f"""You are a helpful robot assistant that can control a ROS2-based robot.
                        When asked to navigate somewhere, use the navigate_to_location tool with the location name and coordinates.
                        Available locations:
                        - kitchen (x=0.88, y=-3.45)
                        - origin (x=0.0, y=0.0)
                        - office (x=-3.9, y=-3.55)
                        

                        """
            )
        ]
    ),
    "response_modalities": ["TEXT"],
    "tools": tools
}

class GeminiTextToTextNode(Node):
    def __init__(self):
        super().__init__('gemini_text_to_text_node')
        self.get_logger().info('Gemini Text-to-Text Node started')
        
        # Create the action client to send navigation goals
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.get_logger().info('Waiting for Nav2 action server...')
        self.wait_for_nav_server()
        
    def wait_for_nav_server(self):
        # Wait for the action server to be available
        server_ready = self.nav_to_pose_client.wait_for_server(timeout_sec=5.0)
        if server_ready:
            self.get_logger().info('Nav2 action server is available!')
        else:
            self.get_logger().warn('Nav2 action server not available after waiting. Will try again when needed.')
        return server_ready
            
    def send_nav_goal(self, x, y, yaw=0.0):
        # Create a NavigateToPose action goal
        goal_msg = NavigateToPose.Goal()
        
        # Build the pose
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        
        # Set position
        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)
        goal_msg.pose.pose.position.z = 0.0
        
        # Set orientation using quaternion from yaw angle
        goal_msg.pose.pose.orientation.x = 0.0
        goal_msg.pose.pose.orientation.y = 0.0
        goal_msg.pose.pose.orientation.z = math.sin(yaw/2)
        goal_msg.pose.pose.orientation.w = math.cos(yaw/2)
        
        self.get_logger().info(f'Sending navigation goal to x={x}, y={y}, yaw={yaw}')
        
        # Send the goal
        self.nav_to_pose_client.wait_for_server()
        self._send_goal_future = self.nav_to_pose_client.send_goal_async(
            goal_msg, 
            feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)
        return True
        
    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected by Nav2 action server')
            return
        
        self.get_logger().info('Goal accepted by Nav2 action server')
        
        # Request the result
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.result_callback)
    
    def result_callback(self, future):
        result = future.result().result
        status = future.result().status
        if status == 4:  # SUCCEEDED
            self.get_logger().info('Navigation succeeded')
        else:
            self.get_logger().info(f'Navigation finished with status {status}')
    
    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Distance remaining: {feedback.distance_remaining}')
        
    def get_tool_response(self, tool_name, parameters):
        if tool_name == "navigate_to_location":
            location = parameters.get("location", "unknown")
            x = parameters.get("x", 0.0)
            y = parameters.get("y", 0.0)
            
            # Log the navigation command
            self.get_logger().info(f'Navigating to {location} at coordinates ({x}, {y})')
            
            # Check if Nav2 action server is available, wait if not
            if not self.nav_to_pose_client.server_is_ready():
                server_ready = self.wait_for_nav_server()
                if not server_ready:
                    return f"Navigation to {location} failed: Nav2 action server not available."
            
            # Send the actual navigation goal
            success = self.send_nav_goal(x, y)
            
            if success:
                return f"Navigation initiated to {location} at coordinates ({x}, {y})."
            else:
                return f"Failed to send navigation goal to {location}."
        else:
            return "Unknown tool."

    async def run_session(self):
        # Connect to the API
        async with client.aio.live.connect(model=model, config=config) as session:
            while rclpy.ok():
                # Process any pending callbacks
                rclpy.spin_once(self, timeout_sec=0.1)
                
                message = input("User> ")
                if message.lower() == "exit":
                    break

                await session.send_client_content(
                    turns={"role": "user", "parts": [{"text": message}]},
                    turn_complete=True
                )

                # Process the response from the model
                async for chunk in session.receive():
                    # Check if the chunk is a server content or a tool call

                    # Print the server content
                    if chunk.server_content:
                        if chunk.text is not None:
                            print(chunk.text, end="")

                    # Print the tool call
                    elif chunk.tool_call:
                        print(f"\nTool call received")
                        function_responses = []
                        for fc in chunk.tool_call.function_calls:
                            # Print details about the function call
                            print(f"Function: {fc.name}, ID: {fc.id}")
                            print(f"Arguments: {fc.args}")
                            
                            # Process the tool call and get result
                            result = self.get_tool_response(fc.name, fc.args)
                            
                            function_response = types.FunctionResponse(
                                id=fc.id,
                                name=fc.name,
                                response={"result": result}
                            )
                            function_responses.append(function_response)

                        # Send all function responses back to the model
                        await session.send_tool_response(function_responses=function_responses)


def main(args=None):
    rclpy.init(args=args)
    node = GeminiTextToTextNode()
    
    try:
        asyncio.run(node.run_session())
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Shutting down Gemini Text-to-Text Node')
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
