#include <rclcpp/rclcpp.hpp>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/task_constructor/task.h>
#include <moveit/task_constructor/solvers.h>
#include <moveit/task_constructor/stages.h>

static const rclcpp::Logger LOGGER = rclcpp::get_logger("can_pick");
namespace mtc = moveit::task_constructor;

class CanPickNode {
public:
    CanPickNode(const rclcpp::NodeOptions& options);

    rclcpp::node_interfaces::NodeBaseInterface::SharedPtr getNodeBaseInterface();

    void doTask();

private:
    mtc::Task createTask();
    mtc::Task task_;
    rclcpp::Node::SharedPtr node_;
};

rclcpp::node_interfaces::NodeBaseInterface::SharedPtr CanPickNode::getNodeBaseInterface() {
    return node_->get_node_base_interface();
}

CanPickNode::CanPickNode(const rclcpp::NodeOptions& options)
  : node_{ std::make_shared<rclcpp::Node>("can_pick_node", options) }
{
}

void CanPickNode::doTask() {
    task_ = createTask();

    try
    {
      task_.init();
    }
    catch (mtc::InitStageException& e)
    {
      RCLCPP_ERROR_STREAM(LOGGER, e);
      return;
    }

    if (!task_.plan(5))
    {
      RCLCPP_ERROR_STREAM(LOGGER, "Task planning failed");
      return;
    }
    task_.introspection().publishSolution(*task_.solutions().front());

    auto result = task_.execute(*task_.solutions().front());
    if (result.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS)
    {
      RCLCPP_ERROR_STREAM(LOGGER, "Task execution failed");
      return;
    }

    return;
}

mtc::Task CanPickNode::createTask() {
    mtc::Task task;
    task.stages()->setName("can pick task");
    task.loadRobotModel(node_);

    const auto& arm_group_name = "arm";
    const auto& hand_group_name = "gripper";
    const auto& hand_frame = "end_effector_frame";

    // Set task properties
    task.setProperty("group", arm_group_name);
    task.setProperty("eef", hand_group_name);
    task.setProperty("ik_frame", hand_frame);

    // Sampling planner
    auto sampling_planner = std::make_shared<mtc::solvers::PipelinePlanner>(node_);
    auto interpolation_planner = std::make_shared<mtc::solvers::JointInterpolationPlanner>();

    // Cartesian planner
    auto cartesian_planner = std::make_shared<mtc::solvers::CartesianPath>();
    cartesian_planner->setMaxVelocityScalingFactor(1.0);
    cartesian_planner->setMaxAccelerationScalingFactor(1.0);
    cartesian_planner->setStepSize(.01);

    /****************************************************
     *               Current State                      *
     ****************************************************/
    {
        auto stage = std::make_unique<mtc::stages::CurrentState>("current");
        task.add(std::move(stage));
    }

    /****************************************************
     *               Move to Ready                      *
     ****************************************************/
    {
        auto stage = std::make_unique<mtc::stages::MoveTo>("move to ready", interpolation_planner);
        stage->setGroup(arm_group_name);
        stage->setGoal("ready");
        task.add(std::move(stage));
    }

    /****************************************************
     *               Open Gripper                       *
     ****************************************************/
    {
        auto stage = std::make_unique<mtc::stages::MoveTo>("open gripper", interpolation_planner);
        stage->setGroup(hand_group_name);
        stage->setGoal("open");
        task.add(std::move(stage));
    }

    /****************************************************
     *       Move to Target (relative to depth cam)    *
     ****************************************************/
    {
        auto stage = std::make_unique<mtc::stages::MoveTo>("move to target", sampling_planner);
        stage->setGroup(arm_group_name);
        stage->setIKFrame(hand_frame);

        // Define target pose relative to depth_cam_optical_link
        geometry_msgs::msg::PoseStamped target_pose;
        target_pose.header.frame_id = "world";

        // Adjust these values based on where you want to reach
        target_pose.pose.position.x = 0.4;  // 30cm in front of camera
        target_pose.pose.position.y = 0.0;  // centered
        target_pose.pose.position.z = 0.1;  // at camera height

        // Orientation (gripper facing down or as needed)
        target_pose.pose.orientation.w = 1.0;
        target_pose.pose.orientation.x = 0.0;
        target_pose.pose.orientation.y = 0.0;
        target_pose.pose.orientation.z = 0.0;

        stage->setGoal(target_pose);
        task.add(std::move(stage));
    }

    /****************************************************
     *               Close Gripper                      *
     ****************************************************/
    {
        auto stage = std::make_unique<mtc::stages::MoveTo>("close gripper", interpolation_planner);
        stage->setGroup(hand_group_name);
        stage->setGoal("close");
        task.add(std::move(stage));
    }

    /****************************************************
     *               Return to Ready                    *
     ****************************************************/
    {
        auto stage = std::make_unique<mtc::stages::MoveTo>("return to ready", interpolation_planner);
        stage->setGroup(arm_group_name);
        stage->setGoal("ready");
        task.add(std::move(stage));
    }

    return task;
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;
    options.automatically_declare_parameters_from_overrides(true);

    auto can_pick_node = std::make_shared<CanPickNode>(options);
    rclcpp::executors::MultiThreadedExecutor executor;

    auto spin_thread = std::make_unique<std::thread>([&executor, &can_pick_node]() {
      executor.add_node(can_pick_node->getNodeBaseInterface());
      executor.spin();
      executor.remove_node(can_pick_node->getNodeBaseInterface());
    });

    can_pick_node->doTask();

    spin_thread->join();
    rclcpp::shutdown();
    return 0;
}
