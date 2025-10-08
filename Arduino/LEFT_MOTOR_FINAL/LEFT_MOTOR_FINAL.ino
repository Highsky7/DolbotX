/**
 * @file LEFT_MOTOR_FINAL.ino
 * @brief Control sketch for the LEFT side motors of the DolbotX robot.
 *
 * This sketch is designed for controlling the three motors on the left side
 * of the robot. It differs from the main wheel control sketch by focusing
 * only on one side and including additional features for robust testing.
 *
 * Features:
 * - Controls three motors based on a single encoder (M1).
 * - Receives target velocity for the left side via serial command ("VL<velocity>\n").
 * - Implements a PID control loop to manage motor speed.
 * - Includes a command timeout (watchdog) to stop motors if no command is received.
 * - Applies an Exponential Moving Average (EMA) filter to the RPM reading for stability.
 *
 * Serial Command Format: "VL<velocity_mps>\n"
 * Example: "VL-0.5\n"
 */

#include <MsTimer2.h>
#include <math.h>
#include <util/atomic.h> // Header for using ATOMIC_BLOCK

// === Pin Definitions ===
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

// === Constants ===
#define PULSES_PER_REV 324
#define CONTROL_MS 20 // Control loop period [ms]
const double dt_s = CONTROL_MS / 1000.0;
const double GEAR_RATIO = 1.0 / 71.0;     // Gear ratio (motor->wheel)
const double WHEEL_DIAM = 0.135;          // Wheel diameter [m]
const double WHEEL_CIRC = M_PI * WHEEL_DIAM; // Wheel circumference [m]

// PID Gains
const float KP = 2.8;
const float KI = 3.4;
const float KD = 0.0;
const double I_MAX = 3000.0;

// === Global Variables ===
volatile long encL = 0;
// Variable to store target velocity from serial communication
float vel_left_Sub = 0.0;
float integral1 = 0.0f, prevErr1 = 0.0f;
volatile bool flag_Control = false;
volatile bool velReceived = false; // Flag to indicate if a command has been received

// ★ ADDED: Watchdog (command timeout) -----------------------
unsigned long lastCmdMs = 0;      // Time of last valid command reception (ms)
const unsigned long CMD_TIMEOUT_MS = 300;  // Timeout threshold
// ----------------------------------------------------

// ★ ADDED: RPM Low-pass filter (EMA) ---------------------------
double wheelRPM_L_filt = 0.0;
const double RPM_ALPHA = 0.3; // 0~1, higher value means faster tracking (more noise)
// ----------------------------------------------------

// Buffer for non-blocking serial reception
const int SERIAL_BUFFER_SIZE = 64;
char serialBuffer[SERIAL_BUFFER_SIZE];
int bufferIndex = 0;

// Interrupts, Timers
void ISR_Left() {
  if (digitalRead(M1_ENC_B) == LOW) encL--;
  else encL++;
}

void onTimer() {
  flag_Control = true;
}

/**
 * @brief Parses serial data (e.g., "VL-0.5\n").
 */
void parseSerial() {
  while (Serial.available() > 0) {
    char incomingChar = Serial.read();

    if (incomingChar == '\n') {
      serialBuffer[bufferIndex] = '\0'; // Null-terminate the string

      // Check if it starts with "VL"
      if (strncmp(serialBuffer, "VL", 2) == 0) {
        // Convert the numeric part after "VL" to float
        vel_left_Sub = atof(serialBuffer + 2);
        velReceived = true; // Valid command received

        // ★ ADDED: Update last reception time for watchdog
        lastCmdMs = millis();

        // Reset PID state to prevent Integral Windup on new command
        integral1 = 0.0f;
        prevErr1 = 0.0f;
      }

      bufferIndex = 0; // Reset buffer
    } else {
      if (bufferIndex < SERIAL_BUFFER_SIZE - 1) {
        serialBuffer[bufferIndex++] = incomingChar;
      } else {
        bufferIndex = 0; // Prevent buffer overflow
      }
    }
  }
}

/**
 * @brief Drives a single motor.
 * @param pwmPin The PWM pin.
 * @param dirPin The direction pin.
 * @param cmd The command value (magnitude for PWM, sign for direction).
 */
void driveMotor(int pwmPin, int dirPin, double cmd) {
  int pwm = (int)round(fabs(cmd));
  if (pwm > 250) pwm = 250;
  if (pwm < 15) pwm = 0;
  digitalWrite(dirPin, (cmd >= 0) ? HIGH : LOW);
  analogWrite(pwmPin, pwm);
}

/**
 * @brief Calculates PID control output.
 * @param target The target setpoint.
 * @param current The current measured value.
 * @param integral Reference to the integral term.
 * @param prevErr Reference to the previous error term.
 * @return The PID command value.
 */
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

  // Enable all drivers
  digitalWrite(start_pin1, HIGH);
  digitalWrite(start_pin2, HIGH);
  digitalWrite(start_pin3, HIGH);

  // Stop motors before starting
  driveMotor(M1_PWM, M1_DIR, 0);

  MsTimer2::set(CONTROL_MS, onTimer);
  MsTimer2::start();
}

void loop() {
  // Continuously check for incoming serial commands
  parseSerial();

  // ★ ADDED: Watchdog (timeout) - Stop if no command received or timeout
  if (!velReceived || (millis() - lastCmdMs) > CMD_TIMEOUT_MS) {
    driveMotor(M1_PWM, M1_DIR, 0);
    driveMotor(M2_PWM, M2_DIR, 0);
    driveMotor(M3_PWM, M3_DIR, 0);
    return;
  }

  // Execute control loop based on timer period
  if (!flag_Control) return;
  flag_Control = false;

  // Safely read the encoder value using an atomic block to prevent interrupt conflicts
  long cntL;
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
    cntL = encL;
    encL = 0;
  }

  // Convert target velocity (m/s) to wheel RPM
  double targetRPM_L = vel_left_Sub * 60.0 / WHEEL_CIRC;

  // Convert encoder pulses to motor RPM
  double motorRPM_L = (cntL / (double)PULSES_PER_REV) * (60.0 / dt_s);

  // Convert motor RPM to wheel RPM
  double wheelRPM_L = motorRPM_L * GEAR_RATIO;

  // Calculate current actual velocity in m/s (for debugging)
  double actual_vel_left = wheelRPM_L * WHEEL_CIRC / 60.0;

  // ★ ADDED: Apply EMA low-pass filter to RPM before using in PID
  wheelRPM_L_filt = RPM_ALPHA * wheelRPM_L + (1.0 - RPM_ALPHA) * wheelRPM_L_filt;

  // Calculate PID control value (using filtered RPM)
  double cmdL = PID(targetRPM_L, wheelRPM_L_filt, integral1, prevErr1);

  // Stop motor if target velocity is very low (Deadzone)
  if (fabs(vel_left_Sub) <= 0.05) {
    cmdL = 0;
    // Reset integral term on stop command to prevent overshoot on next start
    integral1 = 0.0f;
  }

  // Limit PWM output value
  cmdL = constrain(cmdL, -250, 250);

  // Apply control signal to all three motors (M2, M3 are opposite direction)
  driveMotor(M1_PWM, M1_DIR,  cmdL);
  driveMotor(M2_PWM, M2_DIR, -cmdL);
  driveMotor(M3_PWM, M3_DIR, -cmdL);

  // === Serial plotter output for debugging ===
  // Serial.print(vel_left_Sub); Serial.print(" ");
  // Serial.print(actual_vel_left); Serial.print(" ");
  // Serial.println(cmdL);
}