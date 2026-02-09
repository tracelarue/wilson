#include <WiFi.h>
#include <WebServer.h>
#include <ArduinoOTA.h>
#include <ESP32Servo.h>
#include <math.h>
#include "mini_fridge_secrets.h"

WebServer server(80);
Servo servo1;
Servo servo2;

const int SERVO1_PIN = D5;   // starts at 0
const int SERVO2_PIN = D6;   // starts at 180
const int STEP_DELAY_MS = 500;
bool isOpen = false;
bool isBusy = false;
int d6StepDelayMs = 40;       // lower = faster movement for D6
int d6Angle = 180;
bool servosAttached = false;

enum LedMode {
  LED_MODE_OFF,
  LED_MODE_RED,
  LED_MODE_GREEN,
  LED_MODE_BLUE_BREATH
};

void setLed(LedMode mode) {
  int red = HIGH;
  int green = HIGH;
  int blue = HIGH;

  if (mode == LED_MODE_RED) {
    red = LOW;
  } else if (mode == LED_MODE_GREEN) {
    green = LOW;
  } else if (mode == LED_MODE_BLUE_BREATH) {
    float brightness = (exp(sinf(millis() / 1500.0f * PI)) - 0.36787944f) * 108.0f;
    int pwm = 255 - (int)brightness;  // active-low LED
    analogWrite(LEDB, constrain(pwm, 0, 255));
    digitalWrite(LEDR, HIGH);
    digitalWrite(LEDG, HIGH);
    return;
  }

  digitalWrite(LEDR, red);
  digitalWrite(LEDG, green);
  digitalWrite(LEDB, blue);
}

void attachServosIfNeeded() {
  if (servosAttached) {
    return;
  }

  servo1.attach(SERVO1_PIN);
  servo2.attach(SERVO2_PIN);
  servosAttached = true;
}

void detachServosIfNeeded() {
  if (!servosAttached) {
    return;
  }

  servo1.detach();
  servo2.detach();
  servosAttached = false;
}

void openFridge() {
  attachServosIfNeeded();
  isBusy = true;

  setLed(LED_MODE_GREEN);
  Serial.println("SERVO D5 -> 90");
  servo1.write(90);
  delay(STEP_DELAY_MS);

  bool ledOn = false;
  while (d6Angle > 0) {
    d6Angle--;
    Serial.print("SERVO D6 -> ");
    Serial.println(d6Angle);
    servo2.write(d6Angle);
    if (ledOn) {
      setLed(LED_MODE_OFF);
    } else {
      setLed(LED_MODE_GREEN);
    }
    ledOn = !ledOn;
    delay(d6StepDelayMs);
  }

  setLed(LED_MODE_GREEN);
  Serial.println("SERVO D5 -> 0");
  servo1.write(0);
  delay(STEP_DELAY_MS);

  setLed(LED_MODE_OFF);
  isOpen = true;
  isBusy = false;
}

void closeFridge() {
  attachServosIfNeeded();
  isBusy = true;

  bool ledOn = false;
  while (d6Angle < 180) {
    d6Angle++;
    Serial.print("SERVO D6 -> ");
    Serial.println(d6Angle);
    servo2.write(d6Angle);
    if (ledOn) {
      setLed(LED_MODE_OFF);
    } else {
      setLed(LED_MODE_GREEN);
    }
    ledOn = !ledOn;
    delay(d6StepDelayMs);
  }

  setLed(LED_MODE_OFF);
  isOpen = false;
  isBusy = false;
}

void handleCommand() {
  if (!server.hasArg("command")) {
    server.send(400, "text/plain", "missing command");
    return;
  }

  String command = server.arg("command");
  command.toLowerCase();

  if (server.hasArg("d6_step_delay_ms")) {
    int requested = server.arg("d6_step_delay_ms").toInt();
    if (requested < 1 || requested > 100) {
      server.send(400, "text/plain", "bad d6_step_delay_ms (use 1..100)");
      return;
    }
    d6StepDelayMs = requested;
  }

  if (command == "open") {
    Serial.println("COMMAND open");
    openFridge();
    server.send(200, "text/plain", "open");
  } else if (command == "close") {
    Serial.println("COMMAND close");
    closeFridge();
    server.send(200, "text/plain", "close");
  } else if (command == "toggle") {
    Serial.println("COMMAND toggle");
    if (isOpen) {
      closeFridge();
      server.send(200, "text/plain", "close");
    } else {
      openFridge();
      server.send(200, "text/plain", "open");
    }
  } else {
    server.send(400, "text/plain", "bad command");
  }
}

void setup() {
  Serial.begin(115200);

  pinMode(LEDR, OUTPUT);
  pinMode(LEDG, OUTPUT);
  pinMode(LEDB, OUTPUT);
  setLed(LED_MODE_OFF);

  WiFi.begin(MINI_FRIDGE_WIFI_SSID, MINI_FRIDGE_WIFI_PASSWORD);
  bool redOn = false;
  while (WiFi.status() != WL_CONNECTED) {
    if (redOn) {
      setLed(LED_MODE_OFF);
    } else {
      setLed(LED_MODE_RED);
    }
    redOn = !redOn;
    delay(500);
    Serial.print(".");
  }
  setLed(LED_MODE_OFF);

  ArduinoOTA.setHostname(MINI_FRIDGE_OTA_HOSTNAME);
  ArduinoOTA.begin();

  attachServosIfNeeded();
  Serial.println("SERVO D5 -> 0");
  servo1.write(0);
  Serial.println("SERVO D6 -> 180");
  servo2.write(180);
  d6Angle = 180;

  server.on("/mini_fridge", HTTP_GET, handleCommand);
  server.begin();

  setLed(LED_MODE_GREEN);
  delay(1000);
  setLed(LED_MODE_OFF);
  detachServosIfNeeded();

  Serial.println("ready");
  Serial.println(WiFi.localIP());
}

void loop() {
  ArduinoOTA.handle();
  server.handleClient();

  if (!isBusy && WiFi.status() == WL_CONNECTED) {
    setLed(LED_MODE_BLUE_BREATH);
  }

  if (!isBusy) {
    detachServosIfNeeded();
  }
}
