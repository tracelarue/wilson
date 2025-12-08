/***************************************************************
   Motor driver function definitions - by James Nugen
   *************************************************************/

#ifdef L298_MOTOR_DRIVER
  #define RIGHT_MOTOR_BACKWARD 5
  #define LEFT_MOTOR_BACKWARD  6
  #define RIGHT_MOTOR_FORWARD  9
  #define LEFT_MOTOR_FORWARD   10
  #define RIGHT_MOTOR_ENABLE 12
  #define LEFT_MOTOR_ENABLE 13
#endif

#ifdef DROK_L298_MOTOR_DRIVER
  #define RIGHT_MOTOR_BACKWARD 8
  #define RIGHT_MOTOR_FORWARD  9
  #define RIGHT_MOTOR_ENABLE 11 
  
  #define LEFT_MOTOR_BACKWARD  6
  #define LEFT_MOTOR_FORWARD   7
  #define LEFT_MOTOR_ENABLE 10
#endif

#ifdef TB6612FNG_MOTOR_DRIVER
  // Left Motor
  #define LEFT_MOTOR_PWM_PIN 6     // PWM pin for left motor speed
  #define LEFT_MOTOR_DIR_PIN1 11     // Direction pin for left motor
  #define LEFT_MOTOR_DIR_PIN2 12    // Direction pin for left motor
  // Right Motor
  #define RIGHT_MOTOR_PWM_PIN 5  // PWM pin for right motor speed
  #define RIGHT_MOTOR_DIR_PIN1 9    // Direction pin for right motor
  #define RIGHT_MOTOR_DIR_PIN2 8    // Direction pin for right motor
  // Enable
  #define MOTOR_ENABLE 10       // Enable pin for both motors
#endif

void initMotorController();
void setMotorSpeed(int i, int spd);
void setMotorSpeeds(int leftSpeed, int rightSpeed);
