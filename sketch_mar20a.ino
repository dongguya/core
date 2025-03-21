void setup() {
  pinMode(LED_BUILTIN, OUTPUT); // STM32 보드에서는 LED_BUILTIN = PA5
}

void loop() {
  digitalWrite(LED_BUILTIN, HIGH);
  delay(500);
  digitalWrite(LED_BUILTIN, LOW);
  delay(500);
}