#include <Servo.h>
#include <Wire.h>
#include <MLX90393.h>

Servo baseServo, shoulderServo, elbowServo, wristServo, gripperServo;

#define MIN_PULSE_WIDTH 500
#define MAX_PULSE_WIDTH 2500
#define SERVO_PHYSICAL_ANGLE 270
#define MAX_USER_ANGLE 270

// Serial buffer
String inputBuffer = "";

// Current servo positions (tracking) - matches ROS2 "idle" state from SRDF
int currentBase = 135, currentShoulder = 149, currentElbow = 10, currentWrist = 100, currentGripper = 0;
unsigned long lastCmdMs = 0;
bool servosAttached = true;
#define SERVO_IDLE_TIMEOUT_MS 500

// LED breathing variables
#define LED_PIN 11
int ledBrightness = 25;
int ledDirection = 1;
unsigned long lastLedUpdate = 0;
#define LED_UPDATE_INTERVAL 5  // Match arm_control timing

// MLX90393 sensor variables
MLX90393 mlx;
MLX90393::txyz data;
MLX90393::txyzRaw raw_data;

// MLX sensor configuration
// NOTE: Higher OSR/DIG_FILT increases conversion time significantly.
int GAIN = 0;
int RES_X = 0;
int RES_Y = 0;
int RES_Z = 0;
int OSR = 1;       // fastest
int DIG_FILT = 7;  // fastest
int taredZ = 0;

// MLX sensor timing
unsigned long lastMLXUpdate = 0;
#define MLX_UPDATE_INTERVAL 20  // Read every 20ms
bool mlx_inflight = false;
unsigned long mlx_start_ms = 0;
uint16_t mlx_conv_ms = 0;
const uint8_t mlx_flags = MLX90393::X_FLAG | MLX90393::Y_FLAG | MLX90393::Z_FLAG;

void setup() {
  Serial.begin(115200);

  // Initialize MLX sensor
  Wire.begin();
  // Raise I2C clock to improve MLX read throughput (default is 100kHz).
  Wire.setClock(400000);
  delay(50);
  mlx.begin();
  delay(50);
  mlx.setGainSel(GAIN);
  mlx.setResolution(RES_X, RES_Y, RES_Z);
  mlx.setOverSampling(OSR);
  mlx.setDigitalFiltering(DIG_FILT);
  mlx_conv_ms = mlx.convDelayMillis();

  // Setup LED pin
  pinMode(LED_PIN, OUTPUT);

  baseServo.attach(3, MIN_PULSE_WIDTH, MAX_PULSE_WIDTH);
  shoulderServo.attach(5, MIN_PULSE_WIDTH, MAX_PULSE_WIDTH);
  elbowServo.attach(6, MIN_PULSE_WIDTH, MAX_PULSE_WIDTH);
  wristServo.attach(9, MIN_PULSE_WIDTH, MAX_PULSE_WIDTH);
  gripperServo.attach(10, MIN_PULSE_WIDTH, MAX_PULSE_WIDTH);
  servosAttached = true;
  lastCmdMs = millis();

  // Default positions - matches ROS2 "idle" state {0.0, 0.2495, -2.1817, -0.6098, 0.0}
  setServoPosition(baseServo, 135);     // joint_1: 0.0 rad
  setServoPosition(shoulderServo, 149); // joint_2: 0.2495 rad (14.3°)
  setServoPosition(elbowServo, 10);     // joint_3: -2.1817 rad (-125°)
  setServoPosition(wristServo, 100);    // joint_4: -0.6098 rad (-34.9°)
  setServoPosition(gripperServo, 0);    // gripper: 0.0 rad

  Serial.println("ArduinoBot Servo Controller Ready (Direct Mode)");
}

void loop() {
  // Handle incoming servo commands (match arm_control loop ordering)
  while (Serial.available() > 0) {
    char incomingChar = Serial.read();

    if (incomingChar == ',') {
      processCommand(inputBuffer);
      inputBuffer = "";
    } else {
      inputBuffer += incomingChar;
    }
  }

  // Detach servos on idle to reduce jitter
  if (servosAttached && (millis() - lastCmdMs > SERVO_IDLE_TIMEOUT_MS)) {
    baseServo.detach();
    shoulderServo.detach();
    elbowServo.detach();
    wristServo.detach();
    gripperServo.detach();
    servosAttached = false;
  }

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

  // Update MLX sensor reading (non-blocking)
  if (currentMillis - lastMLXUpdate >= MLX_UPDATE_INTERVAL) {
    lastMLXUpdate = currentMillis;
    if (!mlx_inflight) {
      mlx.startMeasurement(mlx_flags);
      mlx_start_ms = currentMillis;
      mlx_inflight = true;
    }
  }

  if (mlx_inflight && (currentMillis - mlx_start_ms >= mlx_conv_ms)) {
    mlx.readMeasurement(mlx_flags, raw_data);
    data = mlx.convertRaw(raw_data);
    taredZ = data.z + 18630;

    // Print exactly 3 values for hardware interface parser
    Serial.print(data.x);
    Serial.print(", ");
    Serial.print(data.y);
    Serial.print(", ");
    Serial.println(taredZ);

    mlx_inflight = false;
  }

  // If a measurement completed, only print when there's space to avoid blocking.
  if (mlx_inflight && (currentMillis - mlx_start_ms >= mlx_conv_ms)) {
    mlx.readMeasurement(mlx_flags, raw_data);
    data = mlx.convertRaw(raw_data);
    taredZ = data.z + 18630;

    // Print exactly 3 values for hardware interface parser
    // Guard against Serial TX blocking which can introduce jitter.
    if (Serial.availableForWrite() >= 32) {
      Serial.print(data.x);
      Serial.print(", ");
      Serial.print(data.y);
      Serial.print(", ");
      Serial.println(taredZ);
    }

    mlx_inflight = false;
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

  // Re-attach on first command after idle and restore current positions
  if (!servosAttached) {
    baseServo.attach(3, MIN_PULSE_WIDTH, MAX_PULSE_WIDTH);
    shoulderServo.attach(5, MIN_PULSE_WIDTH, MAX_PULSE_WIDTH);
    elbowServo.attach(6, MIN_PULSE_WIDTH, MAX_PULSE_WIDTH);
    wristServo.attach(9, MIN_PULSE_WIDTH, MAX_PULSE_WIDTH);
    gripperServo.attach(10, MIN_PULSE_WIDTH, MAX_PULSE_WIDTH);
    servosAttached = true;

    setServoPosition(baseServo, currentBase);
    setServoPosition(shoulderServo, currentShoulder);
    setServoPosition(elbowServo, currentElbow);
    setServoPosition(wristServo, currentWrist);
    setServoPosition(gripperServo, currentGripper);
  }

  lastCmdMs = millis();

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
