#include <MsTimer2.h>
#include <math.h>

// === 핀 정의 (Pin Definitions) ===
#define M1_PWM 2
#define M1_DIR 3
#define M1_ENC_A 18
#define M1_ENC_B 19
#define M2_PWM 12
#define M2_DIR 13
#define M3_PWM 8
#define M3_DIR 9
#define M4_PWM 4
#define M4_DIR 5
#define M4_ENC_A 20
#define M4_ENC_B 21
#define M5_PWM 11
#define M5_DIR 10
#define M6_PWM 6
#define M6_DIR 7
#define start_pin1 22
#define start_pin2 24
#define start_pin3 26
#define start_pin4 34
#define start_pin5 30
#define start_pin6 32

// === 상수 정의 (Constants) ===
#define PULSES_PER_REV 350
#define CONTROL_MS 20 // 제어 주기 [ms]
const double dt_s = CONTROL_MS / 1000.0;
const double GEAR_RATIO = 1.0 / 71.0;     // 기어비 (모터→휠)
const double WHEEL_DIAM = 0.135;          // 휠 지름 [m]
const double WHEEL_CIRC = M_PI * WHEEL_DIAM; // 휠 원주 [m]

// PID 게인
const float KP = 0.8;
const float KI = 0.0;
const float KD = 0.0;
const double I_MAX = 3000.0;

// === 전역 변수 (Global Variables) ===
volatile long encL = 0, encR = 0;
float vel_left_Sub = 0.0, vel_right_Sub = 0.0;
float integral1 = 0.0f, prevErr1 = 0.0f;
float integral4 = 0.0f, prevErr4 = 0.0f;
volatile bool flag_Control = false;
volatile bool velReceived = false;

// 논블로킹 시리얼 수신을 위한 버퍼
const int SERIAL_BUFFER_SIZE = 64;
char serialBuffer[SERIAL_BUFFER_SIZE];
int bufferIndex = 0;

// 인터럽트, 타이머 (Interrupts, Timers)
void ISR_Left() {
  if (digitalRead(M1_ENC_B) == LOW) encL++;
  else encL--;
}
void ISR_Right() {
  if (digitalRead(M4_ENC_B) == LOW) encR++;
  else encR--;
}
void onTimer() {
  flag_Control = true;
}

// *** sscanf 대신 strtok과 atof를 사용하는 안정적인 파싱 함수 ***
void parseSerial() {
  while (Serial.available() > 0) {
    char incomingChar = Serial.read();

    if (incomingChar == '\n') {
      serialBuffer[bufferIndex] = '\0'; 

      if (strncmp(serialBuffer, "VL", 2) == 0) {
        // strtok을 사용하기 위해 버퍼 복사본을 만듭니다.
        char tempBuffer[SERIAL_BUFFER_SIZE];
        strcpy(tempBuffer, serialBuffer);

        char *ptr = strtok(tempBuffer, " "); // 첫 번째 토큰("VL") 분리
        if (ptr != NULL) {
          ptr = strtok(NULL, " "); // 두 번째 토큰(좌측 속도) 분리
          if (ptr != NULL) {
            vel_left_Sub = atof(ptr); // atof: 문자열을 float으로 변환
            
            ptr = strtok(NULL, " "); // 세 번째 토큰(우측 속도) 분리
            if (ptr != NULL) {
              vel_right_Sub = atof(ptr);
              velReceived = true;
            }
          }
        }
      }
      bufferIndex = 0;
    } 
    else {
      if (bufferIndex < SERIAL_BUFFER_SIZE - 1) {
        serialBuffer[bufferIndex++] = incomingChar;
      } else {
        bufferIndex = 0;
      }
    }
  }
}

// 모터 드라이브 (Motor Drive function)
void driveMotor(int pwmPin, int dirPin, double cmd) {
  int pwm = (int)round(fabs(cmd));
  if (pwm > 200) pwm = 200;
  if (pwm < 15) pwm = 0;
  digitalWrite(dirPin, (cmd >= 0) ? HIGH : LOW);
  analogWrite(pwmPin, pwm);
}

// PID 계산 (PID Calculation)
double PID(double target, double current, float& integral, float& prevErr) {
  double error = target - current;
  integral += error * dt_s;
  integral = constrain(integral, -I_MAX, I_MAX);
  double derivative = (error - prevErr) / dt_s;
  prevErr = error;
  return KP * error + KI * integral + KD * derivative;
}

void setup() {
  Serial.begin(57600);

  pinMode(M1_ENC_A, INPUT_PULLUP);
  pinMode(M1_ENC_B, INPUT_PULLUP);
  pinMode(M4_ENC_A, INPUT_PULLUP);
  pinMode(M4_ENC_B, INPUT_PULLUP);
  pinMode(M1_PWM, OUTPUT); pinMode(M1_DIR, OUTPUT);
  pinMode(M4_PWM, OUTPUT); pinMode(M4_DIR, OUTPUT);
  pinMode(M2_PWM, OUTPUT); pinMode(M2_DIR, OUTPUT);
  pinMode(M3_PWM, OUTPUT); pinMode(M3_DIR, OUTPUT);
  pinMode(M5_PWM, OUTPUT); pinMode(M5_DIR, OUTPUT);
  pinMode(M6_PWM, OUTPUT); pinMode(M6_DIR, OUTPUT);
  pinMode(start_pin1, OUTPUT); pinMode(start_pin2, OUTPUT);
  pinMode(start_pin3, OUTPUT); pinMode(start_pin4, OUTPUT);
  pinMode(start_pin5, OUTPUT); pinMode(start_pin6, OUTPUT);
  
  attachInterrupt(digitalPinToInterrupt(M1_ENC_A), ISR_Left, RISING);
  attachInterrupt(digitalPinToInterrupt(M4_ENC_A), ISR_Right, RISING);

  digitalWrite(start_pin1, HIGH); digitalWrite(start_pin2, HIGH);
  digitalWrite(start_pin3, HIGH); digitalWrite(start_pin4, HIGH);
  digitalWrite(start_pin5, HIGH); digitalWrite(start_pin6, HIGH);

  driveMotor(M1_PWM, M1_DIR, 0);
  driveMotor(M4_PWM, M4_DIR, 0);
  
  MsTimer2::set(CONTROL_MS, onTimer);
  MsTimer2::start();
}

void loop() {
  parseSerial();

  if (!velReceived) {
    driveMotor(M1_PWM, M1_DIR, 0); driveMotor(M2_PWM, M2_DIR, 0);
    driveMotor(M3_PWM, M3_DIR, 0); driveMotor(M4_PWM, M4_DIR, 0);
    driveMotor(M5_PWM, M5_DIR, 0); driveMotor(M6_PWM, M6_DIR, 0);
    return;
  }
  
  if (!flag_Control) return;
  flag_Control = false;

  noInterrupts();
  long cntL = encL, cntR = encR;
  encL = encR = 0;
  interrupts();

  double targetRPM_L = vel_left_Sub * 60.0 / WHEEL_CIRC;
  double targetRPM_R = vel_right_Sub * 60.0 / WHEEL_CIRC;

  double motorRPM_L = (cntL / (double)PULSES_PER_REV) * (60.0 / dt_s);
  double motorRPM_R = (cntR / (double)PULSES_PER_REV) * (60.0 / dt_s);

  double wheelRPM_L = motorRPM_L * GEAR_RATIO;
  double wheelRPM_R = motorRPM_R * GEAR_RATIO;

  double cmdL = PID(targetRPM_L, wheelRPM_L, integral1, prevErr1);
  double cmdR = PID(targetRPM_R, wheelRPM_R, integral4, prevErr4);

  if (fabs(vel_left_Sub) <= 0.05) cmdL = 0;
  if (fabs(vel_right_Sub) <= 0.05) cmdR = 0;

  cmdL = constrain(cmdL, -200, 200);
  cmdR = constrain(cmdR, -200, 200);

  driveMotor(M1_PWM, M1_DIR,  cmdL);   // 기준(엔코더 있음)
  driveMotor(M2_PWM, M2_DIR, -cmdL);   // M1과 반대
  driveMotor(M3_PWM, M3_DIR, -cmdL);   // M1과 반대

  driveMotor(M4_PWM, M4_DIR,  cmdR);   // 기준(엔코더 있음)
  driveMotor(M5_PWM, M5_DIR, -cmdR);   // M4와 반대
  driveMotor(M6_PWM, M6_DIR, -cmdR);   // M4와 반대
}