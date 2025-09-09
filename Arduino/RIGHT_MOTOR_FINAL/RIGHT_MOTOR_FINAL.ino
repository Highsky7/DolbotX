#include <MsTimer2.h>
#include <math.h>
#include <util/atomic.h> // ATOMIC_BLOCK 사용을 위해 헤더 파일 추가

// === 핀 정의 (Pin Definitions) ===
// 우측 모터
#define M4_PWM 2
#define M4_DIR 3
#define M4_ENC_A 18
#define M4_ENC_B 19
#define M5_PWM 4
#define M5_DIR 5
#define M6_PWM 6
#define M6_DIR 7
#define start_pin4 28
#define start_pin5 30
#define start_pin6 32

// === 상수 정의 (Constants) ===
#define PULSES_PER_REV 324
#define CONTROL_MS 20 // 제어 주기 [ms]
const double dt_s = CONTROL_MS / 1000.0;
const double GEAR_RATIO = 1.0 / 71.0;     // 기어비 (모터→휠)
const double WHEEL_DIAM = 0.135;          // 휠 지름 [m]
const double WHEEL_CIRC = M_PI * WHEEL_DIAM; // 휠 원주 [m]

// === PID 게인 ===
// 우측 모터 (M4, M5, M6)
const float KP = 3.0;
const float KI = 2.9;
const float KD = 0.0;
const double I_MAX = 3000.0;

// === 전역 변수 (Global Variables) ===
volatile long encR = 0;
// 시리얼 통신으로 목표 속도를 저장할 변수
float vel_right_Sub = 0.0;
float integral_R = 0.0f, prevErr_R = 0.0f;
volatile bool flag_Control = false;
volatile bool velReceived = false; // 명령 수신 여부 플래그

// ★ 추가: 워치독(명령 타임아웃)
unsigned long lastCmdMs = 0;                  // 마지막 유효 명령 수신 시각(ms)
const unsigned long CMD_TIMEOUT_MS = 300;     // 타임아웃 임계값

// ★ 추가: RPM 저역통과(EMA)
double wheelRPM_R_filt = 0.0;
const double RPM_ALPHA = 0.3;                 // 0~1, 클수록 빠른 추종(노이즈↑)

// 논블로킹 시리얼 수신을 위한 버퍼
const int SERIAL_BUFFER_SIZE = 64;
char serialBuffer[SERIAL_BUFFER_SIZE];
int bufferIndex = 0;

// 인터럽트, 타이머 (Interrupts, Timers) - 우측 모터 전용
void ISR_Right() {
  if (digitalRead(M4_ENC_B) == LOW) encR--;
  else encR++;
}

void onTimer() {
  flag_Control = true;
}

// 시리얼 데이터 파싱 함수 (예: "VR 0.5\n")
void parseSerial() {
  while (Serial.available() > 0) {
    char incomingChar = Serial.read();

    if (incomingChar == '\n') {
      serialBuffer[bufferIndex] = '\0'; // 문자열 종료

      // "VR" (Velocity Right)로 시작하는지 확인
      if (strncmp(serialBuffer, "VR", 2) == 0) {
        // "VR" 다음의 숫자 부분을 float으로 변환
        vel_right_Sub = atof(serialBuffer + 2);
        velReceived = true; // 유효한 명령 수신 완료

        // ★ 추가: 워치독 갱신 + PID 상태 리셋
        lastCmdMs = millis();
        integral_R = 0.0f;
        prevErr_R = 0.0f;
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
double PID(double target, double current, float& integral, float& prevErr, const float Kp, const float Ki, const float Kd) {
  double error = target - current;
  integral += error * dt_s;
  integral = constrain(integral, -I_MAX, I_MAX);
  double derivative = (error - prevErr) / dt_s;
  prevErr = error;
  return Kp * error + Ki * integral + Kd * derivative;
}

void setup() {
  Serial.begin(57600);
  
  // 우측 핀 설정
  pinMode(M4_ENC_A, INPUT_PULLUP);
  pinMode(M4_ENC_B, INPUT_PULLUP);
  pinMode(M4_PWM, OUTPUT); pinMode(M4_DIR, OUTPUT);
  pinMode(M5_PWM, OUTPUT); pinMode(M5_DIR, OUTPUT);
  pinMode(M6_PWM, OUTPUT); pinMode(M6_DIR, OUTPUT);
  pinMode(start_pin4, OUTPUT); pinMode(start_pin5, OUTPUT);
  pinMode(start_pin6, OUTPUT);

  // 인터럽트 연결
  attachInterrupt(digitalPinToInterrupt(M4_ENC_A), ISR_Right, RISING);
  
  // 우측 드라이버 활성화
  digitalWrite(start_pin4, HIGH); 
  digitalWrite(start_pin5, HIGH);
  digitalWrite(start_pin6, HIGH);

  // 시작 전 모터 정지
  driveMotor(M4_PWM, M4_DIR, 0);
  
  MsTimer2::set(CONTROL_MS, onTimer);
  MsTimer2::start();
}

void loop() {
  // 시리얼 포트에서 들어오는 명령을 계속 확인
  parseSerial();

  // ★ 추가: 워치독(타임아웃) — 유효 명령 전/타임아웃 시 정지
  if (!velReceived || (millis() - lastCmdMs) > CMD_TIMEOUT_MS) {
    driveMotor(M4_PWM, M4_DIR, 0);
    driveMotor(M5_PWM, M5_DIR, 0);
    driveMotor(M6_PWM, M6_DIR, 0);
    return;
  }
  
  // 타이머 주기에 맞춰 제어 루프 실행
  if (!flag_Control) return;
  flag_Control = false;

  // ATOMIC_BLOCK을 사용하여 인터럽트 충돌 없이 안전하게 변수 읽기
  long cntR;
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
    cntR = encR;
    encR = 0;
  }

  // 목표 속도(m/s)를 바퀴 RPM으로 변환
  double targetRPM_R = vel_right_Sub * 60.0 / WHEEL_CIRC;

  // 엔코더 펄스 값을 모터 RPM으로 변환
  double motorRPM_R = (cntR / (double)PULSES_PER_REV) * (60.0 / dt_s);
  
  // 모터 RPM을 바퀴 RPM으로 변환
  double wheelRPM_R = motorRPM_R * GEAR_RATIO;

  // 현재 실제 속도(m/s) 계산 (디버깅용)
  double actual_vel_right = wheelRPM_R * WHEEL_CIRC / 60.0;

  // ★ 추가: RPM 저역통과(EMA) 적용 후 PID 입력으로 사용
  wheelRPM_R_filt = RPM_ALPHA * wheelRPM_R + (1.0 - RPM_ALPHA) * wheelRPM_R_filt;

  // PID 제어값 계산 (필터된 RPM 사용)
  double cmdR = PID(targetRPM_R, wheelRPM_R_filt, integral_R, prevErr_R, KP, KI, KD);

  // 목표 속도가 매우 낮을 경우 모터 정지 (Deadzone)
  if (fabs(vel_right_Sub) <= 0.05) {
    cmdR = 0;
    // 정지 명령 시 적분항을 리셋하여 다음 출발 시 오버슈팅 방지
    integral_R = 0.0f;
  }

  // PWM 출력값 제한
  cmdR = constrain(cmdR, -250, 250);

  // 우측 모터 3개에 제어 신호 인가
  driveMotor(M4_PWM, M4_DIR,  cmdR);
  driveMotor(M5_PWM, M5_DIR, -cmdR);
  driveMotor(M6_PWM, M6_DIR, -cmdR);

  // === 디버깅을 위한 시리얼 플로터 출력 ===
  // Serial.print(vel_right_Sub); Serial.print(" ");       // 우측 목표 속도
  // Serial.print(actual_vel_right); Serial.print(" ");     // 우측 실제 속도(m/s)
  // Serial.println(cmdR);                                  // 제어 출력(PWM 근사)
}

