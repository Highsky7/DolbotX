#include <SPI.h>
#include <mcp_can.h>

#define Red_Pin 2
#define Green_Pin 3
#define PWM_Pin1 4
#define PWM_Pin2 5

const uint8_t CAN_CS_PIN = 10;
MCP_CAN CAN(CAN_CS_PIN);

static inline void applyCommand(const char* raw) {
  // 소문자 비교
  String s = String(raw); s.toLowerCase();

  if (s == "roka") {
    digitalWrite(Green_Pin, HIGH);
    digitalWrite(Red_Pin, LOW);
  } else if (s == "enemy") {
    digitalWrite(Green_Pin, LOW);
    digitalWrite(Red_Pin, HIGH);
  } else { // "none" 또는 그 외
    digitalWrite(Green_Pin, LOW);
    digitalWrite(Red_Pin, LOW);
  }
}

void setup() {
  pinMode(Red_Pin, OUTPUT);
  pinMode(Green_Pin, OUTPUT);
  pinMode(PWM_Pin1, OUTPUT);
  pinMode(PWM_Pin2, OUTPUT);

  digitalWrite(Red_Pin, LOW);
  digitalWrite(Green_Pin, LOW);

  // PWM은 항상 250 유지
  analogWrite(PWM_Pin1, 250);
  analogWrite(PWM_Pin2, 250);

  Serial.begin(115200);
  delay(200);

  // MCP2515 초기화: 16MHz → 실패 시 8MHz 자동 시도
  byte ok = CAN.begin(MCP_ANY, CAN_500KBPS, MCP_16MHZ);
  if (ok != CAN_OK) ok = CAN.begin(MCP_ANY, CAN_500KBPS, MCP_8MHZ);
  if (ok != CAN_OK) {
    Serial.println("ERROR: MCP2515 init failed");
    while (1) { delay(500); }
  }
  CAN.setMode(MCP_NORMAL);
  Serial.println("CAN ready");
}

void loop() {
  // 보드/타이머 환경에 따라 안전하게 주기적으로 보강
  analogWrite(PWM_Pin1, 250);
  analogWrite(PWM_Pin2, 250);

  // 수신 검사
  if (CAN.checkReceive() == CAN_MSGAVAIL) {
    long unsigned int rxId;
    unsigned char len;
    unsigned char buf[8];

    if (CAN.readMsgBuf(&rxId, &len, buf) == CAN_OK) {
      // ASCII 문자열로 가정 ("ROKA","enemy","none") — 최대 8바이트
      char cmd[9] = {0};
      if (len > 8) len = 8;
      for (byte i = 0; i < len; i++) cmd[i] = (char)buf[i];

      // 개행 제거
      for (byte i = 0; i < len; i++) {
        if (cmd[i] == '\r' || cmd[i] == '\n') { cmd[i] = 0; break; }
      }

      applyCommand(cmd);
      // 디버그
      Serial.print("CAN 0x"); Serial.print(rxId, HEX);
      Serial.print(" cmd=\""); Serial.print(cmd); Serial.println("\"");
    }
  }
}
ㄴ