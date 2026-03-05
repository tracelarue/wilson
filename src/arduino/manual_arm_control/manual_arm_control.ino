/*
 * Manual Arm Control with Preset Pickup Sequence + MLX Streaming
 *
 * Serial protocol:
 *   - Joint commands: b###, s###, e###, w###, g###,
 *   - Pickup command: p001,  (any token beginning with 'p' starts sequence)
 *   - Position query: ?,
 *
 * The pickup sequence below is intentionally easy to tune: edit the
 * five angles (base, shoulder, elbow, wrist, gripper) for each step.
 */

#include <Servo.h>
#include <Wire.h>
#include <MLX90393.h>

Servo baseServo, shoulderServo, elbowServo, wristServo, gripperServo;

#define MIN_PULSE_WIDTH 500
#define MAX_PULSE_WIDTH 2500
#define SERVO_PHYSICAL_ANGLE 270
#define MAX_USER_ANGLE 270

#define ELBOW_OFFSET_DEG 8
#define WRIST_OFFSET_DEG 5

// Serial buffer
String inputBuffer = "";

// Current user-commanded angles (before elbow/wrist offsets)
int currentBase = 135;
int currentShoulder = 149;
int currentElbow = 10;
int currentWrist = 95;
int currentGripper = 0;

unsigned long lastCmdMs = 0;
bool servosAttached = true;
#define SERVO_IDLE_TIMEOUT_MS 800

// LED breathing variables
#define LED_PIN 11
int ledBrightness = 25;
int ledDirection = 1;
unsigned long lastLedUpdate = 0;
#define LED_UPDATE_INTERVAL 5

// MLX90393 sensor variables
MLX90393 mlx;
MLX90393 mlx_ambient;
MLX90393::txyz data;
MLX90393::txyz data_ambient;
MLX90393::txyzRaw raw_data;
MLX90393::txyzRaw raw_data_ambient;

// MLX sensor configuration
int GAIN = 0;
int RES_X = 0;
int RES_Y = 0;
int RES_Z = 0;
int OSR = 2;
int DIG_FILT = 4;
int taredZ = 0;
int taredZAmbient = 0;

// Ambient MLX sensor custom I2C address
#define MLX_AMBIENT_ADDR 1
bool ambient_mlx_ready = true;

// MLX sensor timing
unsigned long lastMLXUpdate = 0;
#define MLX_UPDATE_INTERVAL 20
bool mlx_inflight = false;
unsigned long mlx_start_ms = 0;
uint16_t mlx_conv_ms = 0;
const uint8_t mlx_flags = MLX90393::X_FLAG | MLX90393::Y_FLAG | MLX90393::Z_FLAG;

struct PickupStep {
  int base;
  int shoulder;
  int elbow;
  int wrist;
  int gripper;
  unsigned long holdMs;
};

// ---------------------------------------------------------------------------
// PICKUP SEQUENCE (easy to tune)
// Edit each row's 5 angles (base, shoulder, elbow, wrist, gripper).
// Sequence mirrors grab_object_action stages:
// transition_to_grab -> ready_to_grab -> approach -> close -> lift -> return
// ---------------------------------------------------------------------------
PickupStep pickupSequence[] = {
  // transition_to_grab + hand open
  {152, 135,  58,  45, 220, 650},
  // ready_to_grab + hand open
  {152, 135, 161,  20, 220, 650},
  // approach object (tunable approximation)
  {152, 140, 170,  18, 220, 600},
  // close hand (grab_object)
  {152, 140, 170,  18, 123, 700},
  // lift object
  {152, 126, 138,  52, 123, 700},
  // return ready
  {135, 152,  10, 163, 123, 700},
};

const uint8_t pickupStepCount = sizeof(pickupSequence) / sizeof(pickupSequence[0]);
bool pickupActive = false;
uint8_t pickupIndex = 0;
unsigned long pickupStepStartMs = 0;

void setServoPulseFromUserAngle(Servo &servo, int userAngle, int offsetDeg = 0);
void attachServosIfNeeded();
void applyPose(int base, int shoulder, int elbow, int wrist, int gripper);
void startPickupSequence();
void updatePickupSequence(unsigned long nowMs);
void processCommand(String command);
void sendCurrentPositions();

void setup() {
  Serial.begin(115200);

  // Initialize MLX sensors
  Wire.begin();
  Wire.setClock(400000);
  delay(50);
  mlx.begin();
  delay(50);
  mlx.setGainSel(GAIN);
  mlx.setResolution(RES_X, RES_Y, RES_Z);
  mlx.setOverSampling(OSR);
  mlx.setDigitalFiltering(DIG_FILT);

  mlx_ambient.begin(0, MLX_AMBIENT_ADDR, -1, Wire);
  mlx_ambient.setGainSel(GAIN);
  mlx_ambient.setResolution(RES_X, RES_Y, RES_Z);
  mlx_ambient.setOverSampling(OSR);
  mlx_ambient.setDigitalFiltering(DIG_FILT);

  mlx_conv_ms = mlx.convDelayMillis();

  pinMode(LED_PIN, OUTPUT);

  baseServo.attach(3, MIN_PULSE_WIDTH, MAX_PULSE_WIDTH);
  shoulderServo.attach(5, MIN_PULSE_WIDTH, MAX_PULSE_WIDTH);
  elbowServo.attach(6, MIN_PULSE_WIDTH, MAX_PULSE_WIDTH);
  wristServo.attach(9, MIN_PULSE_WIDTH, MAX_PULSE_WIDTH);
  gripperServo.attach(10, MIN_PULSE_WIDTH, MAX_PULSE_WIDTH);
  servosAttached = true;
  lastCmdMs = millis();

  // Start at idle-like pose
  applyPose(currentBase, currentShoulder, currentElbow, currentWrist, currentGripper);

  Serial.println("Manual Arm Controller Ready");
}

void loop() {
  while (Serial.available() > 0) {
    char incomingChar = Serial.read();
    if (incomingChar == ',') {
      processCommand(inputBuffer);
      inputBuffer = "";
    } else {
      inputBuffer += incomingChar;
    }
  }

  unsigned long currentMillis = millis();

  // Keep sequence non-blocking
  updatePickupSequence(currentMillis);

  // Detach servos on idle to reduce jitter
  if (!pickupActive && servosAttached && (currentMillis - lastCmdMs > SERVO_IDLE_TIMEOUT_MS)) {
    baseServo.detach();
    shoulderServo.detach();
    elbowServo.detach();
    wristServo.detach();
    gripperServo.detach();
    servosAttached = false;
  }

  // LED breathing
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

  // Start MLX conversion periodically
  if (currentMillis - lastMLXUpdate >= MLX_UPDATE_INTERVAL) {
    lastMLXUpdate = currentMillis;
    if (!mlx_inflight) {
      mlx.startMeasurement(mlx_flags);
      if (ambient_mlx_ready) {
        mlx_ambient.startMeasurement(mlx_flags);
      }
      mlx_start_ms = currentMillis;
      mlx_inflight = true;
    }
  }

  // Read and stream MLX samples when ready
  if (mlx_inflight && (currentMillis - mlx_start_ms >= mlx_conv_ms)) {
    mlx.readMeasurement(mlx_flags, raw_data);
    data = mlx.convertRaw(raw_data);
    taredZ = data.z + 18630;

    if (ambient_mlx_ready) {
      mlx_ambient.readMeasurement(mlx_flags, raw_data_ambient);
      data_ambient = mlx_ambient.convertRaw(raw_data_ambient);
      taredZAmbient = data_ambient.z + 18630;
    } else {
      data_ambient.x = 0;
      data_ambient.y = 0;
      taredZAmbient = 0;
    }

    // Exactly 6 values for parser: x,y,z,ax,ay,az
    if (Serial.availableForWrite() >= 48) {
      Serial.print(data.x);
      Serial.print(",");
      Serial.print(data.y);
      Serial.print(",");
      Serial.print(taredZ);
      Serial.print(",");
      Serial.print(data_ambient.x);
      Serial.print(",");
      Serial.print(data_ambient.y);
      Serial.print(",");
      Serial.println(taredZAmbient);
    }
    mlx_inflight = false;
  }
}

void processCommand(String command) {
  command.trim();
  if (command.length() == 0) {
    return;
  }

  if (command == "?") {
    sendCurrentPositions();
    return;
  }

  const char opcode = command.charAt(0);

  // Pickup request: accept any payload beginning with 'p' (e.g., p001)
  if (opcode == 'p' || opcode == 'P') {
    startPickupSequence();
    return;
  }

  if (command.length() < 2) {
    return;
  }

  int angle = command.substring(1).toInt();
  if (angle < 0 || angle > MAX_USER_ANGLE) {
    return;
  }

  // Manual joint commands cancel any running sequence
  pickupActive = false;
  attachServosIfNeeded();
  lastCmdMs = millis();

  switch (opcode) {
    case 'b':
      currentBase = angle;
      setServoPulseFromUserAngle(baseServo, currentBase);
      break;
    case 's':
      currentShoulder = angle;
      setServoPulseFromUserAngle(shoulderServo, currentShoulder);
      break;
    case 'e':
      currentElbow = angle;
      setServoPulseFromUserAngle(elbowServo, currentElbow, ELBOW_OFFSET_DEG);
      break;
    case 'w':
      currentWrist = angle;
      setServoPulseFromUserAngle(wristServo, currentWrist, WRIST_OFFSET_DEG);
      break;
    case 'g':
      currentGripper = angle;
      setServoPulseFromUserAngle(gripperServo, currentGripper);
      break;
    default:
      break;
  }
}

void sendCurrentPositions() {
  Serial.print(currentBase);
  Serial.print(",");
  Serial.print(currentShoulder);
  Serial.print(",");
  Serial.print(currentElbow);
  Serial.print(",");
  Serial.print(currentWrist);
  Serial.print(",");
  Serial.println(currentGripper);
}

void attachServosIfNeeded() {
  if (servosAttached) {
    return;
  }
  baseServo.attach(3, MIN_PULSE_WIDTH, MAX_PULSE_WIDTH);
  shoulderServo.attach(5, MIN_PULSE_WIDTH, MAX_PULSE_WIDTH);
  elbowServo.attach(6, MIN_PULSE_WIDTH, MAX_PULSE_WIDTH);
  wristServo.attach(9, MIN_PULSE_WIDTH, MAX_PULSE_WIDTH);
  gripperServo.attach(10, MIN_PULSE_WIDTH, MAX_PULSE_WIDTH);
  servosAttached = true;

  // Restore the current tracked pose after reattach
  applyPose(currentBase, currentShoulder, currentElbow, currentWrist, currentGripper);
}

void applyPose(int base, int shoulder, int elbow, int wrist, int gripper) {
  currentBase = constrain(base, 0, MAX_USER_ANGLE);
  currentShoulder = constrain(shoulder, 0, MAX_USER_ANGLE);
  currentElbow = constrain(elbow, 0, MAX_USER_ANGLE);
  currentWrist = constrain(wrist, 0, MAX_USER_ANGLE);
  currentGripper = constrain(gripper, 0, MAX_USER_ANGLE);

  setServoPulseFromUserAngle(baseServo, currentBase);
  setServoPulseFromUserAngle(shoulderServo, currentShoulder);
  setServoPulseFromUserAngle(elbowServo, currentElbow, ELBOW_OFFSET_DEG);
  setServoPulseFromUserAngle(wristServo, currentWrist, WRIST_OFFSET_DEG);
  setServoPulseFromUserAngle(gripperServo, currentGripper);
}

void startPickupSequence() {
  attachServosIfNeeded();
  pickupActive = true;
  pickupIndex = 0;
  pickupStepStartMs = millis();
  lastCmdMs = pickupStepStartMs;
  applyPose(
    pickupSequence[pickupIndex].base,
    pickupSequence[pickupIndex].shoulder,
    pickupSequence[pickupIndex].elbow,
    pickupSequence[pickupIndex].wrist,
    pickupSequence[pickupIndex].gripper
  );
}

void updatePickupSequence(unsigned long nowMs) {
  if (!pickupActive) {
    return;
  }

  if (pickupIndex >= pickupStepCount) {
    pickupActive = false;
    return;
  }

  if (nowMs - pickupStepStartMs < pickupSequence[pickupIndex].holdMs) {
    return;
  }

  pickupIndex++;
  if (pickupIndex >= pickupStepCount) {
    pickupActive = false;
    lastCmdMs = nowMs;
    return;
  }

  applyPose(
    pickupSequence[pickupIndex].base,
    pickupSequence[pickupIndex].shoulder,
    pickupSequence[pickupIndex].elbow,
    pickupSequence[pickupIndex].wrist,
    pickupSequence[pickupIndex].gripper
  );
  pickupStepStartMs = nowMs;
  lastCmdMs = nowMs;
}

void setServoPulseFromUserAngle(Servo &servo, int userAngle, int offsetDeg) {
  const int adjusted = constrain(userAngle + offsetDeg, 0, MAX_USER_ANGLE);
  const int pulseWidth = map(adjusted, 0, SERVO_PHYSICAL_ANGLE, MIN_PULSE_WIDTH, MAX_PULSE_WIDTH);
  servo.writeMicroseconds(pulseWidth);
}
