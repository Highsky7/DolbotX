#include <Arduino.h>

// ===== 핀 설정 =====
#define Red_Pin    2
#define Green_Pin  3

// ===== 시리얼 버퍼 =====
#define SERIAL_BUFFER_SIZE 64
static char    serialBuffer[SERIAL_BUFFER_SIZE];
static uint8_t bufferIndex = 0;

// 앞/뒤 공백 제거
static void trim_in_place(char* s) {
  // 뒤쪽 공백 제거
  int end = strlen(s) - 1;
  while (end >= 0 && (s[end] == ' ' || s[end] == '\t' || s[end] == '\r' || s[end] == '\n')) {
    s[end--] = '\0';
  }
  // 앞쪽 공백 제거 (포인터 쉬프트 없이 제자리에서)
  int start = 0;
  while (s[start] == ' ' || s[start] == '\t') start++;
  if (start > 0) {
    int i = 0;
    while (s[start]) s[i++] = s[start++];
    s[i] = '\0';
  }
}

// 소문자 변환
static void tolower_in_place(char* s) {
  for (; *s; ++s) {
    if (*s >= 'A' && *s <= 'Z') *s = *s - 'A' + 'a';
  }
}

// 명령 적용
static void applyCommand(const char* cmd) {
  if (strcmp(cmd, "roka") == 0) {
    digitalWrite(Green_Pin, HIGH);
    digitalWrite(Red_Pin, LOW);
    Serial.println(F("LED: GREEN (roka)"));
  } else if (strcmp(cmd, "enemy") == 0) {
    digitalWrite(Green_Pin, LOW);
    digitalWrite(Red_Pin, HIGH);
    Serial.println(F("LED: RED (enemy)"));
  } else { // "none" 또는 그 외 문자열
    digitalWrite(Green_Pin, LOW);
    digitalWrite(Red_Pin, LOW);
    Serial.println(F("LED: OFF (none/unknown)"));
  }
}

// 시리얼 파싱: 한 줄 명령("roka"/"enemy"/"none")
static void parseSerial() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();

    // CRLF 대응: '\r' 무시
    if (c == '\r') continue;

    if (c == '\n') {
      // 라인 종료 → 문자열 완성
      serialBuffer[bufferIndex] = '\0';
      bufferIndex = 0;

      // 전처리: 공백 제거 + 소문자화
      trim_in_place(serialBuffer);
      tolower_in_place(serialBuffer);

      if (serialBuffer[0] != '\0') {
        applyCommand(serialBuffer);
      }
    } else {
      // 버퍼에 축적
      if (bufferIndex < (SERIAL_BUFFER_SIZE - 1)) {
        serialBuffer[bufferIndex++] = c;
      } else {
        // 오버플로 방지: 라인 버리고 재시작
        bufferIndex = 0;
      }
    }
  }
}

void setup() {
  pinMode(Red_Pin, OUTPUT);
  pinMode(Green_Pin, OUTPUT);
  digitalWrite(Red_Pin, LOW);
  digitalWrite(Green_Pin, LOW);

  Serial.begin(115200);
  // while (!Serial) { ; } // 필요 시 활성화
  delay(100);
  Serial.println(F("SERIAL ready (send: roka | enemy | none)"));
}

void loop() {
  parseSerial();
}

