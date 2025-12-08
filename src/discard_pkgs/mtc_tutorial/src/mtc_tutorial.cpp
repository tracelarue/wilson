#include <rclcpp/rclcpp.hpp>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/task_constructor/task.h>
#include <moveit/task_constructor/solvers.h>
#include <moveit/task_constructor/stages.h>

static const rclcpp::Logger LOGGER = rclcpp::get_logger("mtc_tutorial");
namespace mtc = moveit::task_constructor;

class MTCTaskNode {
public:
    MTCTaskNode(const rclcpp::NodeOptions& options);

    rclcpp::node_interfaces::NodeBaseInterface::SharedPtr getNodeBaseInterface();

    void doTask();

private:
    // Compose an MTC task from a series of stages.
    mtc::Task createTask();
    mtc::Task task_;
    rclcpp::Node::SharedPtr node_;
};

rclcpp::node_interfaces::NodeBaseInterface::SharedPtr MTCTaskNode::getNodeBaseInterface() {
    return node_->get_node_base_interface();
}

MTCTaskNode::MTCTaskNode(const rclcpp::NodeOptions& options)
  : node_{ std::make_shared<rclcpp::Node>("mtc_node", options) }
{
}

void MTCTaskNode::doTask() {
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

mtc::Task MTCTaskNode::createTask() {
    mtc::Task task;
    task.stages()->setName("wilson simple motion task");
    task.loadRobotModel(node_);

    const auto& arm_group_name = "arm";
    const auto& hand_group_name = "gripper";
    const auto& hand_frame = "end_effector_frame";

    // Set task properties
    task.setProperty("group", arm_group_name);
    task.setProperty("eef", hand_group_name);
    task.setProperty("ik_frame", hand_frame);

    // Planners
    auto sampling_planner = std::make_shared<mtc::solvers::PipelinePlanner>(node_);
    auto interpolation_planner = std::make_shared<mtc::solvers::JointInterpolationPlanner>();

    // Cartesian planner
    auto cartesian_planner = std::make_shared<mtc::solvers::CartesianPath>();
    cartesian_planner->setMaxVelocityScalingFactor(1.0);
    cartesian_planner->setMaxAccelerationScalingFactor(1.0);
    cartesian_planner->setStepSize(.01);

    /****************************************************
	 *                                                  *
	 *               Current State                      *
	 *                                                  *
	 ****************************************************/
    {
        auto stage_state_current = std::make_unique<mtc::stages::CurrentState>("current");
        task.add(std::move(stage_state_current));
    }

    /****************************************************
	 *                                                  *
	 *               Move to Ready                      *
	 *                                                  *
	 ****************************************************/
    {
        auto stage = std::make_unique<mtc::stages::MoveTo>("move to ready", interpolation_planner);
        stage->properties().configureInitFrom(mtc::Stage::PARENT, { "group" });
        stage->setGoal("ready");
        task.add(std::move(stage));
    }

    /****************************************************
	 *                                                  *
	 *               Open Gripper                       *
	 *                                                  *
	 ****************************************************/
    {
        auto stage_open_hand =
            std::make_unique<mtc::stages::MoveTo>("open gripper", interpolation_planner);
        stage_open_hand->setGroup(hand_group_name);
        stage_open_hand->setGoal("open");
        task.add(std::move(stage_open_hand));
    }

    /****************************************************
	 *                                                  *
	 *               Move to Target Point               *
	 *                                                  *
	 ****************************************************/
    {
        auto stage = std::make_unique<mtc::stages::MoveTo>("move to target", sampling_planner);
        stage->properties().configureInitFrom(mtc::Stage::PARENT, { "group" });

        // Set target pose
        geometry_msgs::msg::PoseStamped target_pose;
        target_pose.header.frame_id = "world";
        target_pose.pose.position.x = 0.4;
        target_pose.pose.position.y = 0.1;
        target_pose.pose.position.z = 0.2;
        target_pose.pose.orientation.w = 1.0;  // Identity quaternion

        stage->setGoal(target_pose);
        task.add(std::move(stage));
    }

    /****************************************************
	 *                                                  *
	 *               Close Gripper                      *
	 *                                                  *
	 ****************************************************/
    {
        auto stage = std::make_unique<mtc::stages::MoveTo>("close gripper", interpolation_planner);
        stage->setGroup(hand_group_name);
        stage->setGoal("close");
        task.add(std::move(stage));
    }

    /****************************************************
	 *                                                  *
	 *               Return to Ready                    *
	 *                                                  *
	 ****************************************************/
    {
        auto stage = std::make_unique<mtc::stages::MoveTo>("return to ready", interpolation_planner);
        stage->properties().configureInitFrom(mtc::Stage::PARENT, { "group" });
        stage->setGoal("ready");
        task.add(std::move(stage));
    }

    return task;
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;
    options.automatically_declare_parameters_from_overrides(true);

    auto mtc_task_node = std::make_shared<MTCTaskNode>(options);
    rclcpp::executors::MultiThreadedExecutor executor;

    auto spin_thread = std::make_unique<std::thread>([&executor, &mtc_task_node]() {
      executor.add_node(mtc_task_node->getNodeBaseInterface());
      executor.spin();
      executor.remove_node(mtc_task_node->getNodeBaseInterface());
    });

    mtc_task_node->doTask();

    spin_thread->join();
    rclcpp::shutdown();
    return 0;
}
