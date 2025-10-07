#include <Servo.h>

Servo baseServo, shoulderServo, elbowServo, wristServo, gripperServo;

#define MIN_PULSE_WIDTH 500
#define MAX_PULSE_WIDTH 2500
#define SERVO_PHYSICAL_ANGLE 270
#define MAX_USER_ANGLE 270

// Serial buffer
String inputBuffer = "";

// Current servo positions (tracking) - matches ROS2 "idle" state from SRDF
int currentBase = 135, currentShoulder = 149, currentElbow = 0, currentWrist = 100, currentGripper = 0;

// LED breathing variables
#define LED_PIN 11
int ledBrightness = 25;
int ledDirection = 1;
unsigned long lastLedUpdate = 0;
#define LED_UPDATE_INTERVAL 5  // Update every 20ms for smooth breathing

void setup() {
  Serial.begin(115200);

  // Setup LED pin
  pinMode(LED_PIN, OUTPUT);

  baseServo.attach(3, MIN_PULSE_WIDTH, MAX_PULSE_WIDTH);
  shoulderServo.attach(5, MIN_PULSE_WIDTH, MAX_PULSE_WIDTH);
  elbowServo.attach(6, MIN_PULSE_WIDTH, MAX_PULSE_WIDTH);
  wristServo.attach(9, MIN_PULSE_WIDTH, MAX_PULSE_WIDTH);
  gripperServo.attach(10, MIN_PULSE_WIDTH, MAX_PULSE_WIDTH);

  // Default positions - matches ROS2 "idle" state {0.0, 0.2495, -2.3562, -0.6098, 0.0}
  setServoPosition(baseServo, 135);     // joint_1: 0.0 rad
  setServoPosition(shoulderServo, 149); // joint_2: 0.2495 rad (14.3°)
  setServoPosition(elbowServo, 0);      // joint_3: -2.3562 rad (-135°)
  setServoPosition(wristServo, 100);    // joint_4: -0.6098 rad (-34.9°)
  setServoPosition(gripperServo, 0);    // gripper: 0.0 rad

  Serial.println("ArduinoBot Servo Controller Ready (Direct Mode)");
}

void loop() {
  // Update LED breathing effect
  unsigned long currentMillis = millis();
  if (currentMillis - lastLedUpdate >= LED_UPDATE_INTERVAL) {
    lastLedUpdate = currentMillis;
    
    ledBrightness += ledDirection;
    if (ledBrightness >= 255) {
      ledBrightness = 255;
      ledDirection = -1;
    } else if (ledBrightness <= 25) {
      ledBrightness = 25;
      ledDirection = 1;
    }
    
    analogWrite(LED_PIN, ledBrightness);
  }

  while (Serial.available() > 0) {
    char incomingChar = Serial.read();

    if (incomingChar == ',') {
      processCommand(inputBuffer);
      inputBuffer = "";
    } else {
      inputBuffer += incomingChar;
    }
  }
}

void processCommand(String command) {
  if (command.length() < 4) return;

  char joint = command.charAt(0);
  
  // Handle position read request
  if (command == "?") {
    // Send current positions: base,shoulder,elbow,wrist,gripper
    Serial.print(currentBase);
    Serial.print(",");
    Serial.print(currentShoulder);
    Serial.print(",");
    Serial.print(currentElbow);
    Serial.print(",");
    Serial.print(currentWrist);
    Serial.print(",");
    Serial.print(currentGripper);
    Serial.println();
    return;
  }
  
  int angle = command.substring(1).toInt();

  if (angle < 0 || angle > MAX_USER_ANGLE) return;

  switch (joint) {
    case 'b': 
      setServoPosition(baseServo, angle); 
      currentBase = angle;
      break;
    case 's': 
      setServoPosition(shoulderServo, angle); 
      currentShoulder = angle;
      break;
    case 'e': 
      setServoPosition(elbowServo, angle); 
      currentElbow = angle;
      break;
    case 'w': 
      setServoPosition(wristServo, angle); 
      currentWrist = angle;
      break;
    case 'g': 
      setServoPosition(gripperServo, angle); 
      currentGripper = angle;
      break;
    default: break;
  }
}

void setServoPosition(Servo &servo, int angle) {
  angle = constrain(angle, 0, MAX_USER_ANGLE);
  int pulseWidth = map(angle, 0, SERVO_PHYSICAL_ANGLE, MIN_PULSE_WIDTH, MAX_PULSE_WIDTH);
  servo.writeMicroseconds(pulseWidth);
}
