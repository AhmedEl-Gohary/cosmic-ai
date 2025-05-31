#include <LiquidCrystal.h>

LiquidCrystal lcd(12, 11, 5, 4, 3, 2);

int sensorVal=0;

void setup() {
  Serial.begin(9600);
  lcd.begin(16, 2);
}

void loop() {
  int sensorVal = analogRead(A0);

  Serial.print("sensor Value: ");
  Serial.print(sensorVal);

  float voltage = (sensorVal / 1024.0) * 5.0;

  Serial.print(", Volts: ");
  Serial.print(voltage);

  float temperature = (voltage - .5) * 100;

  Serial.print(", degrees C: ");
  Serial.println(temperature);

  lcd.setCursor(0, 0);
  lcd.print("Hello");

  lcd.setCursor(0, 1);
  lcd.print("Paris Temp: ");
  lcd.print(temperature);

  delay(1000);
}



