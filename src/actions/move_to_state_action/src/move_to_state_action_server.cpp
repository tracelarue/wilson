#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/robot_state.h>
#include <algorithm>

#include "move_to_state_action/action/move_to_state.hpp"

static const rclcpp::Logger LOGGER = rclcpp::get_logger("move_to_state_action_server");

class MoveToStateActionServer : public rclcpp::Node {
public:
    using MoveToState = move_to_state_action::action::MoveToState;
    using GoalHandleMoveToState = rclcpp_action::ServerGoalHandle<MoveToState>;

    explicit MoveToStateActionServer(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
        : Node("move_to_state_action_server", options)
    {
        using namespace std::placeholders;

        // Create action server
        action_server_ = rclcpp_action::create_server<MoveToState>(
            this,
            "move_to_state",
            std::bind(&MoveToStateActionServer::handle_goal, this, _1, _2),
            std::bind(&MoveToStateActionServer::handle_cancel, this, _1),
            std::bind(&MoveToStateActionServer::handle_accepted, this, _1));

        RCLCPP_INFO(LOGGER, "MoveToState Action Server started");
    }

private:
    rclcpp_action::Server<MoveToState>::SharedPtr action_server_;

    // Action server callbacks
    rclcpp_action::GoalResponse handle_goal(
        const rclcpp_action::GoalUUID& uuid,
        std::shared_ptr<const MoveToState::Goal> goal)
    {
        (void)uuid;
        RCLCPP_INFO(LOGGER, "Received goal request to move to state: '%s'", goal->statename.c_str());
        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
    }

    rclcpp_action::CancelResponse handle_cancel(
        const std::shared_ptr<GoalHandleMoveToState> goal_handle)
    {
        RCLCPP_INFO(LOGGER, "Received request to cancel goal");
        (void)goal_handle;
        return rclcpp_action::CancelResponse::ACCEPT;
    }

    void handle_accepted(const std::shared_ptr<GoalHandleMoveToState> goal_handle)
    {
        using namespace std::placeholders;
        // Spawn a new thread to avoid blocking the executor
        std::thread{std::bind(&MoveToStateActionServer::execute, this, _1), goal_handle}.detach();
    }

    void execute(const std::shared_ptr<GoalHandleMoveToState> goal_handle)
    {
        RCLCPP_INFO(LOGGER, "Executing goal");
        const auto goal = goal_handle->get_goal();
        auto feedback = std::make_shared<MoveToState::Feedback>();
        auto result = std::make_shared<MoveToState::Result>();

        try {
            // Detect which group this state belongs to
            feedback->currentstatus = "Detecting group for state";
            feedback->progresspercentage = 10.0;
            goal_handle->publish_feedback(feedback);

            std::string group_name = detectGroupForState(goal->statename);

            if (group_name.empty()) {
                result->success = false;
                result->message = "State '" + goal->statename + "' not found in SRDF. Available groups: arm, gripper, arm_gripper";
                RCLCPP_ERROR(LOGGER, "%s", result->message.c_str());
                goal_handle->abort(result);
                return;
            }

            RCLCPP_INFO(LOGGER, "State '%s' belongs to group '%s'",
                        goal->statename.c_str(), group_name.c_str());

            // Create move group interface
            feedback->currentstatus = "Initializing move group: " + group_name;
            feedback->progresspercentage = 20.0;
            goal_handle->publish_feedback(feedback);

            moveit::planning_interface::MoveGroupInterface move_group(shared_from_this(), group_name);

            // Set velocity and acceleration scaling for faster motion
            move_group.setMaxVelocityScalingFactor(1.0);  // 100% of max velocity
            move_group.setMaxAccelerationScalingFactor(1.0);  // 100% of max acceleration

            // Set named target
            feedback->currentstatus = "Setting target state: " + goal->statename;
            feedback->progresspercentage = 30.0;
            goal_handle->publish_feedback(feedback);

            if (!move_group.setNamedTarget(goal->statename)) {
                result->success = false;
                result->message = "Failed to set named target '" + goal->statename + "'";
                RCLCPP_ERROR(LOGGER, "%s", result->message.c_str());
                goal_handle->abort(result);
                return;
            }

            // Plan motion
            feedback->currentstatus = "Planning motion";
            feedback->progresspercentage = 50.0;
            goal_handle->publish_feedback(feedback);

            moveit::planning_interface::MoveGroupInterface::Plan plan;
            moveit::core::MoveItErrorCode plan_result = move_group.plan(plan);

            if (plan_result != moveit::core::MoveItErrorCode::SUCCESS) {
                result->success = false;
                result->message = "Motion planning failed for state '" + goal->statename + "'";
                RCLCPP_ERROR(LOGGER, "%s", result->message.c_str());
                goal_handle->abort(result);
                return;
            }

            // Check for cancellation before executing
            if (goal_handle->is_canceling()) {
                result->success = false;
                result->message = "Goal was cancelled before execution";
                goal_handle->canceled(result);
                RCLCPP_INFO(LOGGER, "Goal cancelled");
                return;
            }

            // Execute motion
            feedback->currentstatus = "Executing motion";
            feedback->progresspercentage = 75.0;
            goal_handle->publish_feedback(feedback);

            moveit::core::MoveItErrorCode execute_result = move_group.execute(plan);

            if (execute_result != moveit::core::MoveItErrorCode::SUCCESS) {
                result->success = false;
                result->message = "Motion execution failed for state '" + goal->statename + "'";
                RCLCPP_ERROR(LOGGER, "%s", result->message.c_str());
                goal_handle->abort(result);
                return;
            }

            // Success
            feedback->currentstatus = "Completed";
            feedback->progresspercentage = 100.0;
            goal_handle->publish_feedback(feedback);

            result->success = true;
            result->message = "Successfully moved to state '" + goal->statename + "'";
            goal_handle->succeed(result);
            RCLCPP_INFO(LOGGER, "Goal succeeded");

        } catch (const std::exception& e) {
            RCLCPP_ERROR(LOGGER, "Exception during execution: %s", e.what());
            result->success = false;
            result->message = std::string("Exception: ") + e.what();
            goal_handle->abort(result);
        }
    }

    std::string detectGroupForState(const std::string& state_name)
    {
        // Load robot model to access SRDF group states
        robot_model_loader::RobotModelLoader robot_model_loader(shared_from_this(), "robot_description");
        const moveit::core::RobotModelPtr& robot_model = robot_model_loader.getModel();

        if (!robot_model) {
            RCLCPP_ERROR(LOGGER, "Failed to load robot model");
            return "";
        }

        // Check each joint model group for this state
        const std::vector<std::string> group_names = robot_model->getJointModelGroupNames();

        for (const auto& group_name : group_names) {
            const moveit::core::JointModelGroup* jmg = robot_model->getJointModelGroup(group_name);
            if (jmg) {
                // Get list of named states for this group
                const std::vector<std::string>& default_state_names = jmg->getDefaultStateNames();

                // Check if our target state is in this group's named states
                if (std::find(default_state_names.begin(), default_state_names.end(), state_name)
                    != default_state_names.end()) {
                    return group_name;
                }
            }
        }

        RCLCPP_WARN(LOGGER, "State '%s' not found in any group. Available groups: %s",
                    state_name.c_str(),
                    [&group_names]() {
                        std::string groups_str;
                        for (size_t i = 0; i < group_names.size(); ++i) {
                            groups_str += group_names[i];
                            if (i < group_names.size() - 1) groups_str += ", ";
                        }
                        return groups_str;
                    }().c_str());

        return "";
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;
    options.automatically_declare_parameters_from_overrides(true);

    auto action_server_node = std::make_shared<MoveToStateActionServer>(options);

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(action_server_node);
    executor.spin();

    rclcpp::shutdown();
    return 0;
}
