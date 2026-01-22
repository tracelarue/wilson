#include <Servo.h>

Servo baseServo, shoulderServo, elbowServo, wristServo, gripperServo;

#define MIN_PULSE_WIDTH 500
#define MAX_PULSE_WIDTH 2500
#define SERVO_PHYSICAL_ANGLE 270
#define MAX_USER_ANGLE 270

// Force limiter configuration - using analog pin for force sensor
#define FORCE_SENSOR_PIN A0
long forceLimit = 500;  // Analog reading threshold (0-1023) - adjustable at runtime
bool forceLimitingEnabled = false;

// Serial buffer
String inputBuffer = "";

// Current servo positions (tracking) - matches ROS2 "idle" state from SRDF
int currentBase = 135, currentShoulder = 149, currentElbow = 10, currentWrist = 100, currentGripper = 0;

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

  // Setup force sensor pin (analog input)
  pinMode(FORCE_SENSOR_PIN, INPUT);

  baseServo.attach(3, MIN_PULSE_WIDTH, MAX_PULSE_WIDTH);
  shoulderServo.attach(5, MIN_PULSE_WIDTH, MAX_PULSE_WIDTH);
  elbowServo.attach(6, MIN_PULSE_WIDTH, MAX_PULSE_WIDTH);
  wristServo.attach(9, MIN_PULSE_WIDTH, MAX_PULSE_WIDTH);
  gripperServo.attach(10, MIN_PULSE_WIDTH, MAX_PULSE_WIDTH);

  // Default positions - matches ROS2 "idle" state {0.0, 0.2495, -2.1817, -0.6098, 0.0}
  setServoPosition(baseServo, 135);     // joint_1: 0.0 rad
  setServoPosition(shoulderServo, 149); // joint_2: 0.2495 rad (14.3°)
  setServoPosition(elbowServo, 10);     // joint_3: -2.1817 rad (-125°)
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

  // Handle force limit configuration: f<value>
  if (joint == 'f') {
    long newLimit = command.substring(1).toInt();
    if (newLimit > 0) {
      forceLimit = newLimit;
      forceLimitingEnabled = true;
      Serial.print("Force limit updated to: ");
      Serial.println(forceLimit);
    }
    return;
  }

  // Handle force limiting disable: d
  if (command == "d") {
    forceLimitingEnabled = false;
    Serial.println("Force limiting disabled");
    return;
  }

  // Handle force reading request: r
  if (command == "r") {
    int force = analogRead(FORCE_SENSOR_PIN);
    Serial.print("Current force reading: ");
    Serial.println(force);
    return;
  }

  if (command.length() < 4) return;

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
      angle = angle+7.5; // Elbow offset
      if (angle > MAX_USER_ANGLE) angle = MAX_USER_ANGLE;
      setServoPosition(elbowServo, angle);
      currentElbow = angle;
      break;
    case 'w':
      angle = angle+5; // Wrist offset
      if (angle > MAX_USER_ANGLE) angle = MAX_USER_ANGLE;
      setServoPosition(wristServo, angle);
      currentWrist = angle;
      break;
    case 'g':
      setGripperPositionWithForceLimit(angle);
      break;
    default: break;
  }
}

void setServoPosition(Servo &servo, int angle) {
  angle = constrain(angle, 0, MAX_USER_ANGLE);
  int pulseWidth = map(angle, 0, SERVO_PHYSICAL_ANGLE, MIN_PULSE_WIDTH, MAX_PULSE_WIDTH);
  servo.writeMicroseconds(pulseWidth);
}

void setGripperPositionWithForceLimit(int targetAngle) {
  targetAngle = constrain(targetAngle, 0, MAX_USER_ANGLE);

  // If opening the gripper (target < current), move directly without force check
  if (targetAngle < currentGripper) {
    setServoPosition(gripperServo, targetAngle);
    currentGripper = targetAngle;
    return;
  }

  // If closing the gripper (target >= current), check force if sensor available
  if (sensorAvailable) {
    long force = readForceMagnitude();

    // Check if force limit exceeded
    if (force > forceLimit) {
      Serial.print("Force limit reached! Force: ");
      Serial.print(force);
      Serial.print(" > ");
      Serial.println(forceLimit);
      // Don't move, stay at current position
      return;
    }
  }

  // Force is within limit (or sensor not available), move to target position
  setServoPosition(gripperServo, targetAngle);
  currentGripper = targetAngle;
}

void calibrateForceSensor() {
  Serial.println("Calibrating force sensor... (taking 20 samples)");

  long sumX = 0, sumY = 0, sumZ = 0;
  int samples = 20;

  for(int i = 0; i < samples; i++) {
    mlx.readData(data);
    sumX += data.x;
    sumY += data.y;
    sumZ += data.z;
    delay(50);
  }

  // Calculate average offsets
  offsetX = sumX / samples;
  offsetY = sumY / samples;
  offsetZ = sumZ / samples;
  forceCalibrated = true;

  Serial.print("Force sensor offsets - X: ");
  Serial.print(offsetX);
  Serial.print(", Y: ");
  Serial.print(offsetY);
  Serial.print(", Z: ");
  Serial.println(offsetZ);
}

long readForceMagnitude() {
  if (!forceCalibrated) return 0;

  mlx.readData(data);

  // Apply offsets to get tared values
  long taredX = data.x - offsetX;
  long taredY = data.y - offsetY;
  long taredZ = data.z - offsetZ;

  // Calculate sum of magnitudes (absolute values)
  long sumMagnitudes = abs(taredX) + abs(taredY) + abs(taredZ);

  return sumMagnitudes;
}
