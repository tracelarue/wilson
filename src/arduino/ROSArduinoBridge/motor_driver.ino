/***************************************************************
   Motor driver definitions
   
   Add a "#elif defined" block to this file to include support
   for a particular motor driver.  Then add the appropriate
   #define near the top of the main ROSArduinoBridge.ino file.
   
   *************************************************************/

#ifdef USE_BASE
   
#ifdef POLOLU_VNH5019
  /* Include the Pololu library */
  #include "DualVNH5019MotorShield.h"

  /* Create the motor driver object */
  DualVNH5019MotorShield drive;
  
  /* Wrap the motor driver initialization */
  void initMotorController() {
    drive.init();
  }

  /* Wrap the drive motor set speed function */
  void setMotorSpeed(int i, int spd) {
    if (i == LEFT) drive.setM1Speed(spd);
    else drive.setM2Speed(spd);
  }

  // A convenience function for setting both motor speeds
  void setMotorSpeeds(int leftSpeed, int rightSpeed) {
    setMotorSpeed(LEFT, leftSpeed);
    setMotorSpeed(RIGHT, rightSpeed);
  }
#elif defined POLOLU_MC33926
  /* Include the Pololu library */
  #include "DualMC33926MotorShield.h"

  /* Create the motor driver object */
  DualMC33926MotorShield drive;
  
  /* Wrap the motor driver initialization */
  void initMotorController() {
    drive.init();
  }

  /* Wrap the drive motor set speed function */
  void setMotorSpeed(int i, int spd) {
    if (i == LEFT) drive.setM1Speed(spd);
    else drive.setM2Speed(spd);
  }

  // A convenience function for setting both motor speeds
  void setMotorSpeeds(int leftSpeed, int rightSpeed) {
    setMotorSpeed(LEFT, leftSpeed);
    setMotorSpeed(RIGHT, rightSpeed);
  }
#elif defined L298_MOTOR_DRIVER
  void initMotorController() {
    digitalWrite(RIGHT_MOTOR_ENABLE, HIGH);
    digitalWrite(LEFT_MOTOR_ENABLE, HIGH);
  }
  
  void setMotorSpeed(int i, int spd) {
    unsigned char reverse = 0;
  
    if (spd < 0)
    {
      spd = -spd;
      reverse = 1;
    }
    if (spd > 255)
      spd = 255;
    
    if (i == LEFT) { 
      if      (reverse == 0) { analogWrite(LEFT_MOTOR_FORWARD, spd); analogWrite(LEFT_MOTOR_BACKWARD, 0); }
      else if (reverse == 1) { analogWrite(LEFT_MOTOR_BACKWARD, spd); analogWrite(LEFT_MOTOR_FORWARD, 0); }
    }
    else /*if (i == RIGHT) //no need for condition*/ {
      if      (reverse == 0) { analogWrite(RIGHT_MOTOR_FORWARD, spd); analogWrite(RIGHT_MOTOR_BACKWARD, 0); }
      else if (reverse == 1) { analogWrite(RIGHT_MOTOR_BACKWARD, spd); analogWrite(RIGHT_MOTOR_FORWARD, 0); }
    }
  }
  
  void setMotorSpeeds(int leftSpeed, int rightSpeed) {
    setMotorSpeed(LEFT, leftSpeed);
    setMotorSpeed(RIGHT, rightSpeed);
  }

#elif defined DROK_L298_MOTOR_DRIVER
  void initMotorController() {
    pinMode(RIGHT_MOTOR_FORWARD, OUTPUT);
    pinMode(RIGHT_MOTOR_BACKWARD, OUTPUT);
    pinMode(LEFT_MOTOR_FORWARD, OUTPUT);
    pinMode(LEFT_MOTOR_BACKWARD, OUTPUT);
    pinMode(RIGHT_MOTOR_ENABLE, OUTPUT);
    pinMode(LEFT_MOTOR_ENABLE, OUTPUT);
    
    // Set initial direction pins to LOW
    digitalWrite(RIGHT_MOTOR_FORWARD, LOW);
    digitalWrite(RIGHT_MOTOR_BACKWARD, LOW);
    digitalWrite(LEFT_MOTOR_FORWARD, LOW);
    digitalWrite(LEFT_MOTOR_BACKWARD, LOW);
  }
  
  void setMotorSpeed(int i, int spd) {
    unsigned char reverse = 0;
  
    if (spd < 0)
    {
      spd = -spd;
      reverse = 1;
    }
    if (spd > 255)
      spd = 255;
    
    if (i == LEFT) { 
      if (spd == 0) {
        // Stop the motor
        digitalWrite(LEFT_MOTOR_FORWARD, LOW);
        digitalWrite(LEFT_MOTOR_BACKWARD, LOW);
        analogWrite(LEFT_MOTOR_ENABLE, 0);
      }
      else if (reverse == 0) { 
        // Forward
        digitalWrite(LEFT_MOTOR_FORWARD, HIGH); 
        digitalWrite(LEFT_MOTOR_BACKWARD, LOW); 
        analogWrite(LEFT_MOTOR_ENABLE, spd);
      }
      else if (reverse == 1) { 
        // Backward
        digitalWrite(LEFT_MOTOR_BACKWARD, HIGH); 
        digitalWrite(LEFT_MOTOR_FORWARD, LOW); 
        analogWrite(LEFT_MOTOR_ENABLE, spd);
      }
    }
    else /*if (i == RIGHT) //no need for condition*/ {
      if (spd == 0) {
        // Stop the motor
        digitalWrite(RIGHT_MOTOR_FORWARD, LOW);
        digitalWrite(RIGHT_MOTOR_BACKWARD, LOW);
        analogWrite(RIGHT_MOTOR_ENABLE, 0);
      }
      else if (reverse == 0) { 
        // Forward
        digitalWrite(RIGHT_MOTOR_FORWARD, HIGH); 
        digitalWrite(RIGHT_MOTOR_BACKWARD, LOW); 
        analogWrite(RIGHT_MOTOR_ENABLE, spd);
      }
      else if (reverse == 1) { 
        // Backward
        digitalWrite(RIGHT_MOTOR_BACKWARD, HIGH); 
        digitalWrite(RIGHT_MOTOR_FORWARD, LOW); 
        analogWrite(RIGHT_MOTOR_ENABLE, spd);
      }
    }
  }
  
  void setMotorSpeeds(int leftSpeed, int rightSpeed) {
    setMotorSpeed(LEFT, leftSpeed);
    setMotorSpeed(RIGHT, rightSpeed);
  }
  
#elif defined TB6612FNG_MOTOR_DRIVER

  void initMotorController() {
    digitalWrite(MOTOR_ENABLE, HIGH);
  }
  
  void setMotorSpeed(int i, int spd) {
    unsigned char reverse = 0;
  
    if (spd < 0) {
      spd = -spd;
      reverse = 1;
    }

    if (spd > 255)
      spd = 255;
    
    if (i == LEFT) {
      if (reverse == 0) {
        digitalWrite(LEFT_MOTOR_DIR_PIN1, HIGH);
        digitalWrite(LEFT_MOTOR_DIR_PIN2, LOW);
        analogWrite(LEFT_MOTOR_PWM_PIN, spd);
      } else {
        digitalWrite(LEFT_MOTOR_DIR_PIN1, LOW);
        digitalWrite(LEFT_MOTOR_DIR_PIN2, HIGH);
        analogWrite(LEFT_MOTOR_PWM_PIN, spd);
      }
    } else { // RIGHT motor
      if (reverse == 0) {
        digitalWrite(RIGHT_MOTOR_DIR_PIN1, HIGH);
        digitalWrite(RIGHT_MOTOR_DIR_PIN2, LOW);
        analogWrite(RIGHT_MOTOR_PWM_PIN, spd);
      } else {
        digitalWrite(RIGHT_MOTOR_DIR_PIN1, LOW);
        digitalWrite(RIGHT_MOTOR_DIR_PIN2, HIGH);
        analogWrite(RIGHT_MOTOR_PWM_PIN, spd);
      }
    }
  }
  
  void setMotorSpeeds(int leftSpeed, int rightSpeed) {
    setMotorSpeed(LEFT, leftSpeed);
    setMotorSpeed(RIGHT, rightSpeed);
  }
#else
  #error A motor driver must be selected!
#endif

#endif
