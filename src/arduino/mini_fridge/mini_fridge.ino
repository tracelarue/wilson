#include <Servo.h>

const int RECV_PIN = 3;
const int SERVO_1_PIN = 5;
const int SERVO_2_PIN = 6;

const int SERVO_1_CLOSED_ANGLE = 0;
const int SERVO_1_OPEN_ANGLE = 90;
const int SERVO_2_OPEN_ANGLE = 0;
const int SERVO_2_CLOSED_ANGLE = 180;
const int SERVO_2_STEP_DELAY_MS = 10;
const unsigned long SIGNAL_PULSE_THRESHOLD = 50UL;

const unsigned long DETECTION_WINDOW_MS = 200UL;
const unsigned long PULSE_SAMPLE_STEP_US = 100UL;

unsigned long windowStartMs = 0;
unsigned long pulseCountInWindow = 0;
int lastIrState = HIGH;
bool signalLatched = false;

Servo servo1;
Servo servo2;
int servo1Angle = SERVO_1_CLOSED_ANGLE;
int servo2Angle = SERVO_2_CLOSED_ANGLE;
bool doorIsOpen = false;

void sampleIrPulses() {
  int irState = digitalRead(RECV_PIN);
  // Count active-low pulse starts (falling edges).
  if ((lastIrState == HIGH) && (irState == LOW)) {
    pulseCountInWindow++;
  }
  lastIrState = irState;
}

void moveServoSmooth(Servo &servo, int &currentAngle, int targetAngle, int stepDelayMs) {
  if (currentAngle == targetAngle) {
    return;
  }

  int step = (targetAngle > currentAngle) ? 1 : -1;
  while (currentAngle != targetAngle) {
    currentAngle += step;
    servo.write(currentAngle);
    delay(stepDelayMs);
  }
}

bool isAtClosedState() {
  return (servo1Angle == SERVO_1_CLOSED_ANGLE) && (servo2Angle == SERVO_2_CLOSED_ANGLE);
}

bool isAtOpenState() {
  return (servo1Angle == SERVO_1_CLOSED_ANGLE) && (servo2Angle == SERVO_2_OPEN_ANGLE);
}

void runDoorOpenSequence() {
  servo1Angle = SERVO_1_OPEN_ANGLE;
  servo1.write(servo1Angle);
  delay(250);

  moveServoSmooth(servo2, servo2Angle, SERVO_2_OPEN_ANGLE, SERVO_2_STEP_DELAY_MS);

  servo1Angle = SERVO_1_CLOSED_ANGLE;
  servo1.write(servo1Angle);
  delay(250);

  doorIsOpen = true;
}

void runDoorCloseSequence() {
  moveServoSmooth(servo2, servo2Angle, SERVO_2_CLOSED_ANGLE, SERVO_2_STEP_DELAY_MS);
  doorIsOpen = false;
}

void handleSignal() {
  if (!doorIsOpen && isAtClosedState()) {
    Serial.println("Action: OPENING door");
    runDoorOpenSequence();
    return;
  }

  if (doorIsOpen && isAtOpenState()) {
    Serial.println("Action: CLOSING door");
    runDoorCloseSequence();
    return;
  }
}

void setup() {
  Serial.begin(9600);
  pinMode(RECV_PIN, INPUT_PULLUP);
  lastIrState = digitalRead(RECV_PIN);

  servo1.attach(SERVO_1_PIN);
  servo2.attach(SERVO_2_PIN);
  servo1.write(SERVO_1_CLOSED_ANGLE);
  servo2.write(SERVO_2_CLOSED_ANGLE);

  windowStartMs = millis();
}

void loop() {
  sampleIrPulses();
  delayMicroseconds(PULSE_SAMPLE_STEP_US);

  unsigned long nowMs = millis();
  if ((nowMs - windowStartMs) >= DETECTION_WINDOW_MS) {
    Serial.print("Pulses/200ms: ");
    Serial.println(pulseCountInWindow);

    if (pulseCountInWindow > SIGNAL_PULSE_THRESHOLD) {
      if (!signalLatched) {
        handleSignal();
        signalLatched = true;
      }
    } else {
      signalLatched = false;
    }

    pulseCountInWindow = 0;
    windowStartMs = nowMs;
  }
}
