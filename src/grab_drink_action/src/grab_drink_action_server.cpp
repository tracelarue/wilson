#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/task_constructor/task.h>
#include <moveit/task_constructor/solvers.h>
#include <moveit/task_constructor/stages.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_eigen/tf2_eigen.hpp>

#include "grab_drink_action/action/grab_drink.hpp"

static const rclcpp::Logger LOGGER = rclcpp::get_logger("grab_drink_action_server");
namespace mtc = moveit::task_constructor;

class GrabDrinkActionServer : public rclcpp::Node {
public:
    using GrabDrink = grab_drink_action::action::GrabDrink;
    using GoalHandleGrabDrink = rclcpp_action::ServerGoalHandle<GrabDrink>;

    explicit GrabDrinkActionServer(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
        : Node("grab_drink_action_server", options)
    {
        using namespace std::placeholders;

        // Initialize TF2
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // Create action server
        action_server_ = rclcpp_action::create_server<GrabDrink>(
            this,
            "grab_drink",
            std::bind(&GrabDrinkActionServer::handle_goal, this, _1, _2),
            std::bind(&GrabDrinkActionServer::handle_cancel, this, _1),
            std::bind(&GrabDrinkActionServer::handle_accepted, this, _1));

        RCLCPP_INFO(LOGGER, "Grab Drink Action Server started");
    }

private:
    rclcpp_action::Server<GrabDrink>::SharedPtr action_server_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    // Action server callbacks
    rclcpp_action::GoalResponse handle_goal(
        const rclcpp_action::GoalUUID& uuid,
        std::shared_ptr<const GrabDrink::Goal> goal)
    {
        (void)uuid;
        RCLCPP_INFO(LOGGER, "Received goal request for drink at position [%.3f, %.3f, %.3f] in depth_camera_link_optical frame",
                    goal->targetx, goal->targety, goal->targetz);
        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
    }

    rclcpp_action::CancelResponse handle_cancel(
        const std::shared_ptr<GoalHandleGrabDrink> goal_handle)
    {
        RCLCPP_INFO(LOGGER, "Received request to cancel goal");
        (void)goal_handle;
        return rclcpp_action::CancelResponse::ACCEPT;
    }

    void handle_accepted(const std::shared_ptr<GoalHandleGrabDrink> goal_handle)
    {
        using namespace std::placeholders;
        // Spawn a new thread to avoid blocking the executor
        std::thread{std::bind(&GrabDrinkActionServer::execute, this, _1), goal_handle}.detach();
    }

    void execute(const std::shared_ptr<GoalHandleGrabDrink> goal_handle)
    {
        RCLCPP_INFO(LOGGER, "Executing goal");
        const auto goal = goal_handle->get_goal();
        auto feedback = std::make_shared<GrabDrink::Feedback>();
        auto result = std::make_shared<GrabDrink::Result>();

        try {
            // Transform target position to world frame
            feedback->current_stage = "Transforming coordinates";
            feedback->progress_percentage = 5.0;
            goal_handle->publish_feedback(feedback);

            geometry_msgs::msg::PointStamped target_point_in, target_point_out;
            target_point_in.header.frame_id = "depth_camera_link_optical";
            target_point_in.header.stamp = this->now();
            target_point_in.point.x = goal->targetx;
            target_point_in.point.y = goal->targety;
            target_point_in.point.z = goal->targetz;

            // Wait for transform
            if (!tf_buffer_->canTransform("base_link", "depth_camera_link_optical", tf2::TimePointZero,
                                          tf2::durationFromSec(5.0))) {
                RCLCPP_ERROR(LOGGER, "Cannot transform from 'depth_camera_link_optical' to 'base_link'");
                result->success = false;
                result->message = "Transform unavailable from depth_camera_link_optical to base_link";
                goal_handle->abort(result);
                return;
            }

            target_point_out = tf_buffer_->transform(target_point_in, "base_link");

            RCLCPP_INFO(LOGGER, "Transformed position to base_link frame: [%.3f, %.3f, %.3f]",
                        target_point_out.point.x, target_point_out.point.y, target_point_out.point.z);

            // Set up planning scene with drink at transformed position
            feedback->current_stage = "Setting up planning scene";
            feedback->progress_percentage = 10.0;
            goal_handle->publish_feedback(feedback);

            setupPlanningScene(target_point_out.point);

            // Create and execute MTC task
            feedback->current_stage = "Planning grab task";
            feedback->progress_percentage = 20.0;
            goal_handle->publish_feedback(feedback);

            mtc::Task task = createTask(target_point_out.point, goal_handle, feedback);

            // Initialize task
            feedback->current_stage = "Initializing task";
            feedback->progress_percentage = 30.0;
            goal_handle->publish_feedback(feedback);

            task.init();

            // Plan task
            feedback->current_stage = "Planning motion";
            feedback->progress_percentage = 40.0;
            goal_handle->publish_feedback(feedback);

            if (!task.plan(5)) {
                RCLCPP_ERROR(LOGGER, "Task planning failed");
                result->success = false;
                result->message = "Motion planning failed";
                goal_handle->abort(result);
                return;
            }

            // Publish solution for visualization
            task.introspection().publishSolution(*task.solutions().front());

            // Execute task using MoveGroupInterface
            feedback->current_stage = "Executing motion";
            feedback->progress_percentage = 60.0;
            goal_handle->publish_feedback(feedback);

            // Create move group interfaces for execution
            moveit::planning_interface::MoveGroupInterface move_group(shared_from_this(), "arm");
            moveit::planning_interface::MoveGroupInterface gripper_group(shared_from_this(), "gripper");

            // Execute the solution
            auto execute_result = executeTaskSolution(*task.solutions().front(), move_group, gripper_group);
            if (execute_result.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS) {
                RCLCPP_ERROR(LOGGER, "Task execution failed with error code: %d", execute_result.val);
                result->success = false;
                result->message = "Motion execution failed";
                goal_handle->abort(result);
                return;
            }

            // Success
            feedback->current_stage = "Completed";
            feedback->progress_percentage = 100.0;
            goal_handle->publish_feedback(feedback);

            result->success = true;
            result->message = "Successfully grabbed drink";
            result->final_drink_pose.position.x = target_point_out.point.x;
            result->final_drink_pose.position.y = target_point_out.point.y;
            result->final_drink_pose.position.z = target_point_out.point.z;
            result->final_drink_pose.orientation.w = 1.0;

            goal_handle->succeed(result);
            RCLCPP_INFO(LOGGER, "Goal succeeded");

        } catch (const std::exception& e) {
            RCLCPP_ERROR(LOGGER, "Exception during execution: %s", e.what());
            result->success = false;
            result->message = std::string("Exception: ") + e.what();
            goal_handle->abort(result);
        }
    }

    void setupPlanningScene(const geometry_msgs::msg::Point& position)
    {
        // Hardcoded drink dimensions (standard can size)
        const float drink_height = 0.122;
        const float drink_radius = 0.033;

        moveit_msgs::msg::CollisionObject object;
        object.id = "drink";
        object.header.frame_id = "base_link";
        object.primitives.resize(1);
        object.primitives[0].type = shape_msgs::msg::SolidPrimitive::CYLINDER;
        object.primitives[0].dimensions = { drink_height, drink_radius };

        geometry_msgs::msg::Pose pose;
        pose.position = position;
        pose.orientation.w = 1.0;
        object.pose = pose;

        moveit::planning_interface::PlanningSceneInterface psi;
        psi.applyCollisionObject(object);

        RCLCPP_INFO(LOGGER, "Added drink collision object at [%.3f, %.3f, %.3f] with height %.3f and radius %.3f",
                    position.x, position.y, position.z, drink_height, drink_radius);
    }

    geometry_msgs::msg::Vector3Stamped computeApproachVector(const geometry_msgs::msg::Point& drink_position)
    {
        geometry_msgs::msg::Vector3Stamped approach_vec;
        approach_vec.header.frame_id = "base_link";

        try {
            // Wait for transform to be available
            if (!tf_buffer_->canTransform("base_link", "link_1", tf2::TimePointZero,
                                          tf2::durationFromSec(1.0))) {
                RCLCPP_WARN(LOGGER, "Cannot get transform from link_1 to base_link, using default X direction");
                approach_vec.vector.x = 1.0;
                approach_vec.vector.y = 0.0;
                approach_vec.vector.z = 0.0;
                return approach_vec;
            }

            // Get link_1 position in base_link frame
            geometry_msgs::msg::TransformStamped transform =
                tf_buffer_->lookupTransform("base_link", "link_1", tf2::TimePointZero);

            // Compute vector from link_1 to drink
            double dx = drink_position.x - transform.transform.translation.x;
            double dy = drink_position.y - transform.transform.translation.y;
            double dz = drink_position.z - transform.transform.translation.z;

            // Normalize the vector (ignore Z component for planar approach)
            double length = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (length > 1e-6) {
                approach_vec.vector.x = dx / length;
                approach_vec.vector.y = dy / length;
                approach_vec.vector.z = 0.0;
            } else {
                RCLCPP_WARN(LOGGER, "Base link and drink are at same position, using default X direction");
                approach_vec.vector.x = 1.0;
                approach_vec.vector.y = 0.0;
                approach_vec.vector.z = 0.0;
            }

            RCLCPP_INFO(LOGGER, "Computed approach vector: [%.3f, %.3f, %.3f]",
                        approach_vec.vector.x, approach_vec.vector.y, approach_vec.vector.z);

        } catch (tf2::TransformException& ex) {
            RCLCPP_ERROR(LOGGER, "TF2 exception: %s", ex.what());
            // Default to X direction if transform fails
            approach_vec.vector.x = 1.0;
            approach_vec.vector.y = 0.0;
            approach_vec.vector.z = 0.0;
        }

        return approach_vec;
    }

    // Helper function to recursively extract and execute all trajectories from a solution
    moveit::core::MoveItErrorCode executeTaskSolution(
        const mtc::SolutionBase& solution,
        moveit::planning_interface::MoveGroupInterface& move_group,
        moveit::planning_interface::MoveGroupInterface& gripper)
    {
        // Check if this is a SubTrajectory (leaf node)
        if (const auto* sub_traj = dynamic_cast<const mtc::SubTrajectory*>(&solution)) {
            auto traj = sub_traj->trajectory();
            if (!traj || traj->empty()) {
                return moveit::core::MoveItErrorCode::SUCCESS;  // Empty trajectory, skip
            }

            // Determine which group this trajectory belongs to
            const auto& joint_names = traj->getFirstWayPoint().getVariableNames();
            bool is_gripper = false;
            for (const auto& name : joint_names) {
                if (name.find("gripper") != std::string::npos) {
                    is_gripper = true;
                    break;
                }
            }

            // Execute using appropriate move group
            moveit::planning_interface::MoveGroupInterface::Plan plan;
            robot_trajectory::RobotTrajectory non_const_traj(*traj);  // Create non-const copy
            non_const_traj.getRobotTrajectoryMsg(plan.trajectory_);

            moveit::core::MoveItErrorCode result;
            if (is_gripper) {
                RCLCPP_INFO(LOGGER, "Executing gripper trajectory");
                result = gripper.execute(plan);
            } else {
                RCLCPP_INFO(LOGGER, "Executing arm trajectory");
                result = move_group.execute(plan);
            }

            if (result.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS) {
                RCLCPP_ERROR(LOGGER, "Trajectory execution failed with code: %d", result.val);
                return result;
            }
        }
        // Check if this is a SolutionSequence (container node)
        else if (const auto* sequence = dynamic_cast<const mtc::SolutionSequence*>(&solution)) {
            // Recursively execute all sub-solutions in order
            for (const auto* sub_solution : sequence->solutions()) {
                auto result = executeTaskSolution(*sub_solution, move_group, gripper);
                if (result.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS) {
                    return result;
                }
            }
        }
        // Check if this is a WrappedSolution
        else if (const auto* wrapped = dynamic_cast<const mtc::WrappedSolution*>(&solution)) {
            return executeTaskSolution(*wrapped->wrapped(), move_group, gripper);
        }

        return moveit::core::MoveItErrorCode::SUCCESS;
    }

    mtc::Task createTask(const geometry_msgs::msg::Point& drink_position,
                        [[maybe_unused]] const std::shared_ptr<GoalHandleGrabDrink>& goal_handle,
                        [[maybe_unused]] std::shared_ptr<GrabDrink::Feedback>& feedback)
    {
        mtc::Task task;
        task.stages()->setName("grab drink task");
        task.loadRobotModel(shared_from_this());

        const auto& arm_group_name = "arm";
        const auto& hand_group_name = "gripper";
        const auto& hand_frame = "end_effector_frame";

        // Set task properties
        task.setProperty("group", arm_group_name);
        task.setProperty("eef", hand_group_name);
        task.setProperty("ik_frame", hand_frame);

        // Sampling planner
        auto sampling_planner = std::make_shared<mtc::solvers::PipelinePlanner>(shared_from_this());
        auto interpolation_planner = std::make_shared<mtc::solvers::JointInterpolationPlanner>();

        // Cartesian planner
        auto cartesian_planner = std::make_shared<mtc::solvers::CartesianPath>();
        cartesian_planner->setMaxVelocityScalingFactor(1.0);
        cartesian_planner->setMaxAccelerationScalingFactor(1.0);
        cartesian_planner->setStepSize(.001);

        // Stages
        mtc::Stage* current_state_ptr = nullptr;

        // Current State
        {
            auto stage_state_current = std::make_unique<mtc::stages::CurrentState>("current");
            current_state_ptr = stage_state_current.get();
            task.add(std::move(stage_state_current));
        }

        // Open Hand
        {
            auto stage_open_hand =
                std::make_unique<mtc::stages::MoveTo>("open hand", interpolation_planner);
            stage_open_hand->setGroup(hand_group_name);
            stage_open_hand->setGoal("open");
            task.add(std::move(stage_open_hand));
        }

        // Move to Grab
        {
            auto stage_move_to_grab = std::make_unique<mtc::stages::Connect>(
                "move to grab",
                mtc::stages::Connect::GroupPlannerVector{ { arm_group_name, sampling_planner } });
            stage_move_to_grab->setTimeout(5.0);
            stage_move_to_grab->properties().configureInitFrom(mtc::Stage::PARENT);
            task.add(std::move(stage_move_to_grab));
        }

        // Grab Drink
        {
            auto grasp = std::make_unique<mtc::SerialContainer>("grab drink");
            task.properties().exposeTo(grasp->properties(), { "eef", "group", "ik_frame" });
            grasp->properties().configureInitFrom(mtc::Stage::PARENT, { "eef", "group", "ik_frame" });

            // Approach Drink
            {
                auto stage =
                    std::make_unique<mtc::stages::MoveRelative>("approach drink", cartesian_planner);
                stage->properties().set("marker_ns", "approach_drink");
                stage->properties().set("link", hand_frame);
                stage->properties().configureInitFrom(mtc::Stage::PARENT, { "group" });
                stage->setMinMaxDistance(0.01, 0.03);

                // Compute approach direction dynamically
                geometry_msgs::msg::Vector3Stamped vec = computeApproachVector(drink_position);
                stage->setDirection(vec);
                grasp->insert(std::move(stage));
            }

            // Generate Grasp Pose
            {
                auto stage = std::make_unique<mtc::stages::GenerateGraspPose>("generate grasp pose");
                stage->properties().configureInitFrom(mtc::Stage::PARENT);
                stage->properties().set("marker_ns", "grasp_pose");
                stage->setPreGraspPose("open");
                stage->setObject("drink");
                stage->setAngleDelta(M_PI / 12);
                stage->setMonitoredStage(current_state_ptr);

                // Grasp frame transform
                Eigen::Isometry3d grasp_frame_transform;
                Eigen::Quaterniond q = Eigen::AngleAxisd(0, Eigen::Vector3d::UnitX()) *
                                      Eigen::AngleAxisd(0, Eigen::Vector3d::UnitY()) *
                                      Eigen::AngleAxisd(0, Eigen::Vector3d::UnitZ());
                grasp_frame_transform.linear() = q.matrix();
                grasp_frame_transform.translation().x() = 0.02;
                grasp_frame_transform.translation().y() = 0.0;
                grasp_frame_transform.translation().z() = 0.0;

                // Compute IK
                auto wrapper =
                    std::make_unique<mtc::stages::ComputeIK>("grasp pose IK", std::move(stage));
                wrapper->setMaxIKSolutions(8);
                wrapper->setMinSolutionDistance(1.0);
                wrapper->setIKFrame(grasp_frame_transform, hand_frame);
                wrapper->properties().configureInitFrom(mtc::Stage::PARENT, { "eef", "group" });
                wrapper->properties().configureInitFrom(mtc::Stage::INTERFACE, { "target_pose" });
                grasp->insert(std::move(wrapper));
            }

            // Allow Collision (hand, drink)
            {
                auto stage = std::make_unique<mtc::stages::ModifyPlanningScene>("allow collision (hand,drink)");
                stage->allowCollisions("drink",
                    task.getRobotModel()->getJointModelGroup(hand_group_name)->getLinkModelNamesWithCollisionGeometry(),
                    true);
                grasp->insert(std::move(stage));
            }

            // Close Hand
            {
                auto stage = std::make_unique<mtc::stages::MoveTo>("close hand", interpolation_planner);
                stage->setGroup(hand_group_name);
                stage->setGoal("close");
                grasp->insert(std::move(stage));
            }

            // Attach Drink
            {
                auto stage = std::make_unique<mtc::stages::ModifyPlanningScene>("attach drink");
                stage->attachObject("drink", hand_frame);
                grasp->insert(std::move(stage));
            }

            // Lift Drink
            {
                auto stage = std::make_unique<mtc::stages::MoveRelative>("lift drink", cartesian_planner);
                stage->properties().configureInitFrom(mtc::Stage::PARENT, { "group" });
                stage->setMinMaxDistance(0.0, 0.1);
                stage->setIKFrame(hand_frame);
                stage->properties().set("marker_ns", "lift_drink");

                // Set upward direction
                geometry_msgs::msg::Vector3Stamped vec;
                vec.header.frame_id = "base_link";
                vec.vector.z = 1.0;
                stage->setDirection(vec);
                grasp->insert(std::move(stage));
            }

            task.add(std::move(grasp));
        }

        // Return Home
        {
            auto stage = std::make_unique<mtc::stages::MoveTo>("return home", interpolation_planner);
            stage->properties().configureInitFrom(mtc::Stage::PARENT, { "group" });
            stage->setGoal("ready");
            task.add(std::move(stage));
        }

        return task;
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;
    options.automatically_declare_parameters_from_overrides(true);

    auto action_server_node = std::make_shared<GrabDrinkActionServer>(options);

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(action_server_node);
    executor.spin();

    rclcpp::shutdown();
    return 0;
}
