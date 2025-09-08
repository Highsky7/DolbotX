#include <MsTimer2.h>
#include <math.h>
#include <util/atomic.h> // ATOMIC_BLOCK 사용을 위해 헤더 파일 추가

// === 핀 정의 (Pin Definitions) ===
#define M1_PWM 2
#define M1_DIR 3
#define M1_ENC_A 18
#define M1_ENC_B 19
#define M2_PWM 4
#define M2_DIR 5
#define M3_PWM 6
#define M3_DIR 7
#define start_pin1 28
#define start_pin2 30
#define start_pin3 32

// === 상수 정의 (Constants) ===
#define PULSES_PER_REV 324
#define CONTROL_MS 20 // 제어 주기 [ms]
const double dt_s = CONTROL_MS / 1000.0;
const double GEAR_RATIO = 1.0 / 71.0;     // 기어비 (모터→휠)
const double WHEEL_DIAM = 0.135;          // 휠 지름 [m]
const double WHEEL_CIRC = M_PI * WHEEL_DIAM; // 휠 원주 [m]

// PID 게인
const float KP = 2.7;
const float KI = 3.4;
const float KD = 0.0;
const double I_MAX = 3000.0;

// === 전역 변수 (Global Variables) ===
volatile long encL = 0;
// 시리얼 통신으로 목표 속도를 저장할 변수
float vel_left_Sub = 0.0;
float integral1 = 0.0f, prevErr1 = 0.0f;
volatile bool flag_Control = false;
volatile bool velReceived = false; // 명령 수신 여부 플래그

// 논블로킹 시리얼 수신을 위한 버퍼
const int SERIAL_BUFFER_SIZE = 64;
char serialBuffer[SERIAL_BUFFER_SIZE];
int bufferIndex = 0;

// 인터럽트, 타이머 (Interrupts, Timers)
void ISR_Left() {
  if (digitalRead(M1_ENC_B) == LOW) encL--;
  else encL++;
}

void onTimer() {
  flag_Control = true;
}

// 시리얼 데이터 파싱 함수 (예: "VL-0.5\n")
void parseSerial() {
  while (Serial.available() > 0) {
    char incomingChar = Serial.read();

    if (incomingChar == '\n') {
      serialBuffer[bufferIndex] = '\0'; // 문자열 종료

      // "VL"로 시작하는지 확인
      if (strncmp(serialBuffer, "VL", 2) == 0) {
        // "VL" 다음의 숫자 부분을 float으로 변환
        vel_left_Sub = atof(serialBuffer + 2);
        velReceived = true; // 유효한 명령 수신 완료

        /* === 수정된 부분 START === */
        // 새로운 명령을 받으면 PID 상태를 리셋하여 Integral Windup 방지
        integral1 = 0.0f;
        prevErr1 = 0.0f;
        /* === 수정된 부분 END === */
      }
      
      bufferIndex = 0; // 버퍼 초기화
    } 
    else {
      if (bufferIndex < SERIAL_BUFFER_SIZE - 1) {
        serialBuffer[bufferIndex++] = incomingChar;
      } else {
        bufferIndex = 0; // 버퍼 오버플로우 방지
      }
    }
  }
}

// 모터 드라이브 (Motor Drive function)
void driveMotor(int pwmPin, int dirPin, double cmd) {
  int pwm = (int)round(fabs(cmd));
  if (pwm > 250) pwm = 250;
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
  pinMode(M1_PWM, OUTPUT); pinMode(M1_DIR, OUTPUT);
  pinMode(M2_PWM, OUTPUT); pinMode(M2_DIR, OUTPUT);
  pinMode(M3_PWM, OUTPUT); pinMode(M3_DIR, OUTPUT);
  pinMode(start_pin1, OUTPUT); pinMode(start_pin2, OUTPUT);
  pinMode(start_pin3, OUTPUT);
  
  attachInterrupt(digitalPinToInterrupt(M1_ENC_A), ISR_Left, RISING);
  
  // 모든 드라이버 활성화
  digitalWrite(start_pin1, HIGH); 
  digitalWrite(start_pin2, HIGH);
  digitalWrite(start_pin3, HIGH);

  // 시작 전 모터 정지
  driveMotor(M1_PWM, M1_DIR, 0);
  
  MsTimer2::set(CONTROL_MS, onTimer);
  MsTimer2::start();
}

void loop() {
  // 시리얼 포트에서 들어오는 명령을 계속 확인
  parseSerial();

  // 유효한 속도 명령을 받기 전까지는 모터를 정지
  if (!velReceived) {
    driveMotor(M1_PWM, M1_DIR, 0); 
    driveMotor(M2_PWM, M2_DIR, 0);
    driveMotor(M3_PWM, M3_DIR, 0); 
    return;
  }
  
  // 타이머 주기에 맞춰 제어 루프 실행
  if (!flag_Control) return;
  flag_Control = false;

  // ATOMIC_BLOCK을 사용하여 인터럽트 충돌 없이 안전하게 변수 읽기
  long cntL;
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
    cntL = encL;
    encL = 0;
  }

  // 목표 속도(m/s)를 바퀴 RPM으로 변환
  double targetRPM_L = vel_left_Sub * 60.0 / WHEEL_CIRC;

  // 엔코더 펄스 값을 모터 RPM으로 변환
  double motorRPM_L = (cntL / (double)PULSES_PER_REV) * (60.0 / dt_s);
  
  // 모터 RPM을 바퀴 RPM으로 변환
  double wheelRPM_L = motorRPM_L * GEAR_RATIO;

  // 현재 실제 속도(m/s) 계산 (디버깅용)
  double actual_vel_left = wheelRPM_L * WHEEL_CIRC / 60.0;
  
  // PID 제어값 계산
  double cmdL = PID(targetRPM_L, wheelRPM_L, integral1, prevErr1);

  // 목표 속도가 매우 낮을 경우 모터 정지 (Deadzone)
  if (fabs(vel_left_Sub) <= 0.05) {
    cmdL = 0;
    /* === 수정된 부분 START === */
    // 정지 명령 시 적분항을 리셋하여 다음 출발 시 오버슈팅 방지
    integral1 = 0.0f;
    /* === 수정된 부분 END === */
  }

  // PWM 출력값 제한
  cmdL = constrain(cmdL, -250, 250);

  // 3개의 모터에 제어 신호 인가 (M2, M3는 반대 방향)
  driveMotor(M1_PWM, M1_DIR,  cmdL);
  driveMotor(M2_PWM, M2_DIR, -cmdL);
  driveMotor(M3_PWM, M3_DIR, -cmdL);

  // === 디버깅을 위한 시리얼 플로터 출력 ===
  // Serial.print(vel_left_Sub); // 목표 속도
  // Serial.print(" ");
  // Serial.println(actual_vel_left); // 실제 속도
}