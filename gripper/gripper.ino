int dir_pin = 4;
int pwm_pin = 11;
int test_pin = 13;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  pinMode(dir_pin, OUTPUT);
  pinMode(pwm_pin, OUTPUT);
  pinMode(test_pin, OUTPUT);
  while (!Serial) {
    ; // wait for serial port to connect. Needed for native USB port only
  }
  digitalWrite(dir_pin, HIGH);
}

void loop() {
  // put your main code here, to run repeatedly:

  
  if (Serial.available()) {
    int dir = Serial.read();

    if(dir=='1'){
      digitalWrite(pwm_pin, HIGH);
      digitalWrite(test_pin, HIGH);
    }
    if(dir=='0'){
      digitalWrite(pwm_pin, LOW);
      digitalWrite(test_pin, LOW);
    }

  delay(10);
  }
}
