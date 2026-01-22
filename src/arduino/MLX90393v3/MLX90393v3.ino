#include <Wire.h>
#include <MLX90393.h>

MLX90393 mlx;
MLX90393::txyz data; //Create a structure, called data, of four floats (t, x, y, and z)

int GAIN = 0;
int RES_X = 0;
int RES_Y = 0;
int RES_Z = 0;
int OSR = 2;
int DIG_FILT = 7;

// Store baseline offsets
long offsetX = 0;
long offsetY = 0;
long offsetZ = 0;

byte status;

void setup()
{
  Serial.begin(115200);
  while (!Serial) {
    ; // Wait for serial port to connect (important for Leonardo/Micro)
  }
  delay(500); // Extra settling time after serial connection
  
  Wire.begin();
  delay(50);
  
  Serial.println("Initializing MLX90393...");
  
  mlx.begin(); //iic jumpers set
  delay(50);
  mlx.setGainSel(GAIN);
  mlx.setResolution(RES_X, RES_Y, RES_Z); //x, y, z
  mlx.setOverSampling(OSR);
  mlx.setDigitalFiltering(DIG_FILT);
  
  // Calibrate sensor after everything is stable
  delay(500);
  calibrateSensor();
  
  Serial.println("Ready!");
}

void calibrateSensor() {
  Serial.println("Calibrating... (taking 20 samples)");
  
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
  
  Serial.print("Offsets - X: ");
  Serial.print(offsetX);
  Serial.print(", Y: ");
  Serial.print(offsetY);
  Serial.print(", Z: ");
  Serial.println(offsetZ);
}

void loop()
{
  mlx.readData(data);
  
  // Apply offsets to get tared values
  long taredX = data.x - offsetX;
  long taredY = data.y - offsetY;
  long taredZ = data.z - offsetZ;
  
  // Calculate sum of magnitudes (absolute values)
  long sumMagnitudes = abs(taredX) + abs(taredY) + abs(taredZ);
  
  // Print tared values and sum of magnitudes
  Serial.print(taredX);
  Serial.print(", ");
  Serial.print(taredY);  
  Serial.print(", ");
  Serial.print(taredZ);
  Serial.print(", ");
  Serial.println(sumMagnitudes);
  
  delay(96);
  delayMicroseconds(1000);
}
