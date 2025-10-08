/**
 * @file LED_Control.ino
 * @brief Controls two LEDs (Red and Green) based on serial commands.
 *
 * This sketch listens for specific string commands over the serial port to
 * control a red and a green LED, typically to indicate a status like
 * "friend" or "foe".
 *
 * Serial Commands:
 * - "roka": Turns the Green LED ON and the Red LED OFF.
 * - "enemy": Turns the Red LED ON and the Green LED OFF.
 * - "none" (or any other string): Turns both LEDs OFF.
 *
 * The serial parsing is designed to be robust, handling leading/trailing
 * whitespace and case-insensitivity.
 */

#include <Arduino.h>

// ===== Pin Definitions =====
#define Red_Pin    2  // Pin for the Red LED
#define Green_Pin  3  // Pin for the Green LED

// ===== Serial Buffer =====
#define SERIAL_BUFFER_SIZE 64
static char    serialBuffer[SERIAL_BUFFER_SIZE];
static uint8_t bufferIndex = 0;

/**
 * @brief Trims leading and trailing whitespace from a C-style string in place.
 * @param s The string to trim.
 */
static void trim_in_place(char* s) {
  // Trim trailing whitespace
  int end = strlen(s) - 1;
  while (end >= 0 && (s[end] == ' ' || s[end] == '\t' || s[end] == '\r' || s[end] == '\n')) {
    s[end--] = '\0';
  }
  // Trim leading whitespace (in-place without pointer shift)
  int start = 0;
  while (s[start] == ' ' || s[start] == '\t') start++;
  if (start > 0) {
    int i = 0;
    while (s[start]) s[i++] = s[start++];
    s[i] = '\0';
  }
}

/**
 * @brief Converts a C-style string to lowercase in place.
 * @param s The string to convert.
 */
static void tolower_in_place(char* s) {
  for (; *s; ++s) {
    if (*s >= 'A' && *s <= 'Z') *s = *s - 'A' + 'a';
  }
}

/**
 * @brief Applies the received command to control the LEDs.
 * @param cmd The command string ("roka", "enemy", or "none").
 */
static void applyCommand(const char* cmd) {
  if (strcmp(cmd, "roka") == 0) {
    digitalWrite(Green_Pin, HIGH);
    digitalWrite(Red_Pin, LOW);
  } else if (strcmp(cmd, "enemy") == 0) {
    digitalWrite(Green_Pin, LOW);
    digitalWrite(Red_Pin, HIGH);
  } else { // "none" or any other string
    digitalWrite(Green_Pin, LOW);
    digitalWrite(Red_Pin, LOW);
  }
}

/**
 * @brief Parses incoming serial data line by line.
 * Reads characters into a buffer until a newline is received, then processes the command.
 */
static void parseSerial() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();

    // Handle CRLF: ignore '\r'
    if (c == '\r') continue;

    if (c == '\n') {
      // End of line -> complete the string
      serialBuffer[bufferIndex] = '\0';
      bufferIndex = 0;

      // Preprocess: trim whitespace + convert to lowercase
      trim_in_place(serialBuffer);
      tolower_in_place(serialBuffer);

      if (serialBuffer[0] != '\0') {
        applyCommand(serialBuffer);
      }
    } else {
      // Accumulate in buffer
      if (bufferIndex < (SERIAL_BUFFER_SIZE - 1)) {
        serialBuffer[bufferIndex++] = c;
      } else {
        // Prevent overflow: discard line and restart
        bufferIndex = 0;
      }
    }
  }
}

/**
 * @brief Setup function, runs once at startup.
 * Initializes pin modes and serial communication.
 */
void setup() {
  pinMode(Red_Pin, OUTPUT);
  pinMode(Green_Pin, OUTPUT);
  digitalWrite(Red_Pin, LOW);
  digitalWrite(Green_Pin, LOW);

  Serial.begin(115200);
  // while (!Serial) { ; } // Uncomment if waiting for serial is needed
  delay(100);
  Serial.println(F("SERIAL ready (send: roka | enemy | none)"));
}

/**
 * @brief Main loop. Continuously calls the serial parser.
 */
void loop() {
  parseSerial();
}