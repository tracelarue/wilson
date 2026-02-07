#include <Servo.h>

const int RECV_PIN = 2;
Servo latch; 
Servo door;

int signalCount = 0;
unsigned long windowStart = 0;
bool isDoorOpen = false;

void setup() {
  Serial.begin(9600);
  pinMode(RECV_PIN, INPUT);
  latch.attach(5);
  door.attach(6);
  
  latch.write(0);   // Initial Latch
  door.write(180);  // Initial Door Closed
}

void loop() {
  // IR Receivers are ACTIVE LOW (0 when signal detected)
  if (digitalRead(RECV_PIN) == LOW) {
    unsigned long now = millis();

    // Start a new 1-second window if this is the first pulse
    if (signalCount == 0) {
      windowStart = now;
    }

    // Check if we are still within the 1-second window
    if (now - windowStart <= 1000) {
      signalCount++;
      Serial.print("Signal detected! Count: ");
      Serial.println(signalCount);
      
      // Debounce: Wait for the current IR pulse to end before counting again
      while(digitalRead(RECV_PIN) == LOW); 
    }

    if (signalCount >= 5) {
      handleFridge();
      signalCount = 0; // Reset after action
    }
  }

  // Reset count if 1 second has passed since the first signal
  if (signalCount > 0 && (millis() - windowStart > 1000)) {
    Serial.println("Window expired. Resetting count.");
    signalCount = 0;
  }
}

void handleFridge() {
  if (!isDoorOpen) {
    Serial.println("Action: Opening");
    latch.write(90);
    delay(500);
    // Slow open: 180 to 0
    for(int pos = 180; pos >= 0; pos--) {
      door.write(pos);
      delay(30); // Speed control
    }
    latch.write(0);
    isDoorOpen = true;
  } else {
    Serial.println("Action: Closing");
    // Slow close: 0 to 180
    for(int pos = 0; pos <= 180; pos++) {
      door.write(pos);
      delay(30); // Speed control
    }
    isDoorOpen = false;
  }
  delay(2000); // Cooldown to avoid double-trigger
}