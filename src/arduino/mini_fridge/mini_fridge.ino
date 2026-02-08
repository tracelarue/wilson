#include <WiFi.h>
#include <WebServer.h>
#include <ESP32Servo.h>

// Configure these for your network.
const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

const int SERVO_1_PIN = 18;
const int SERVO_2_PIN = 19;

const int SERVO_1_CLOSED_ANGLE = 0;
const int SERVO_1_OPEN_ANGLE = 90;
const int SERVO_2_OPEN_ANGLE = 0;
const int SERVO_2_CLOSED_ANGLE = 180;
const int SERVO_2_STEP_DELAY_MS = 10;

WebServer server(80);
Servo servo1;
Servo servo2;

int servo1Angle = SERVO_1_CLOSED_ANGLE;
int servo2Angle = SERVO_2_CLOSED_ANGLE;
bool doorIsOpen = false;

void moveServoSmooth(Servo& servo, int& currentAngle, int targetAngle, int stepDelayMs) {
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

String doorStateString() {
  return doorIsOpen ? "open" : "closed";
}

void ensureRequestedState(const String& command) {
  if (command == "open") {
    if (!doorIsOpen || !isAtOpenState()) {
      Serial.println("Action: OPENING door");
      runDoorOpenSequence();
    }
    return;
  }

  if (command == "close") {
    if (doorIsOpen || !isAtClosedState()) {
      Serial.println("Action: CLOSING door");
      runDoorCloseSequence();
    }
    return;
  }

  // toggle
  if (!doorIsOpen && isAtClosedState()) {
    Serial.println("Action: OPENING door (toggle)");
    runDoorOpenSequence();
  } else if (doorIsOpen && isAtOpenState()) {
    Serial.println("Action: CLOSING door (toggle)");
    runDoorCloseSequence();
  }
}

void handleMiniFridgeCommand() {
  String command = "toggle";
  if (server.hasArg("command")) {
    command = server.arg("command");
    command.toLowerCase();
  }

  if (command != "open" && command != "close" && command != "toggle") {
    String message = "invalid command: " + command + " (expected open, close, or toggle)";
    server.send(400, "text/plain", message);
    return;
  }

  ensureRequestedState(command);

  String response = "ok command=" + command + " state=" + doorStateString();
  server.send(200, "text/plain", response);
}

void handleStatus() {
  server.send(200, "text/plain", "state=" + doorStateString());
}

void connectWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.print("Connected. IP address: ");
  Serial.println(WiFi.localIP());
}

void setup() {
  Serial.begin(115200);

  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  servo1.setPeriodHertz(50);
  servo2.setPeriodHertz(50);

  servo1.attach(SERVO_1_PIN, 500, 2400);
  servo2.attach(SERVO_2_PIN, 500, 2400);

  servo1.write(SERVO_1_CLOSED_ANGLE);
  servo2.write(SERVO_2_CLOSED_ANGLE);

  connectWiFi();

  server.on("/mini_fridge", HTTP_GET, handleMiniFridgeCommand);
  server.on("/status", HTTP_GET, handleStatus);
  server.begin();

  Serial.println("Mini fridge HTTP server started");
}

void loop() {
  server.handleClient();
}
