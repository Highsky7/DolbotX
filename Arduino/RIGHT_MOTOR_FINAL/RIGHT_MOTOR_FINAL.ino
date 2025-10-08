/**
 * @file RIGHT_MOTOR_FINAL.ino
 * @brief Control sketch for the RIGHT side motors of the DolbotX robot.
 *
 * This sketch is a counterpart to `LEFT_MOTOR_FINAL.ino` and is designed for
 * controlling the three motors on the right side of the robot.
 *
 * Features:
 * - Controls three motors based on a single encoder (M4).
 * - Receives target velocity for the right side via serial command ("VR<velocity>\n").
 * - Implements a PID control loop to manage motor speed.
 * - Includes a command timeout (watchdog) to stop motors if no command is received.
 * - Applies an Exponential Moving Average (EMA) filter to the RPM reading for stability.
 *
 * Serial Command Format: "VR <velocity_mps>\n"
 * Example: "VR 0.5\n"
 */

#include <MsTimer2.h>
#include <math.h>
#include <util/atomic.h> // Header for using ATOMIC_BLOCK

// === Pin Definitions ===
// Right Side Motors
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

// === Constants ===
#define PULSES_PER_REV 324
#define CONTROL_MS 20 // Control loop period [ms]
const double dt_s = CONTROL_MS / 1000.0;
const double GEAR_RATIO = 1.0 / 71.0;     // Gear ratio (motor->wheel)
const double WHEEL_DIAM = 0.135;          // Wheel diameter [m]
const double WHEEL_CIRC = M_PI * WHEEL_DIAM; // Wheel circumference [m]

// === PID Gains ===
// Right side motors (M4, M5, M6)
const float KP = 3.0;
const float KI = 2.9;
const float KD = 0.0;
const double I_MAX = 3000.0;

// === Global Variables ===
volatile long encR = 0;
// Variable to store target velocity from serial communication
float vel_right_Sub = 0.0;
float integral_R = 0.0f, prevErr_R = 0.0f;
volatile bool flag_Control = false;
volatile bool velReceived = false; // Flag to indicate if a command has been received

// ★ ADDED: Watchdog (command timeout)
unsigned long lastCmdMs = 0;                  // Time of last valid command reception (ms)
const unsigned long CMD_TIMEOUT_MS = 300;     // Timeout threshold

// ★ ADDED: RPM Low-pass filter (EMA)
double wheelRPM_R_filt = 0.0;
const double RPM_ALPHA = 0.3;                 // 0~1, higher value means faster tracking (more noise)

// Buffer for non-blocking serial reception
const int SERIAL_BUFFER_SIZE = 64;
char serialBuffer[SERIAL_BUFFER_SIZE];
int bufferIndex = 0;

// Interrupts, Timers - Right motor only
void ISR_Right() {
  if (digitalRead(M4_ENC_B) == LOW) encR--;
  else encR++;
}

void onTimer() {
  flag_Control = true;
}

/**
 * @brief Parses serial data (e.g., "VR 0.5\n").
 */
void parseSerial() {
  while (Serial.available() > 0) {
    char incomingChar = Serial.read();

    if (incomingChar == '\n') {
      serialBuffer[bufferIndex] = '\0'; // Null-terminate the string

      // Check if it starts with "VR" (Velocity Right)
      if (strncmp(serialBuffer, "VR", 2) == 0) {
        // Convert the numeric part after "VR" to float
        vel_right_Sub = atof(serialBuffer + 2);
        velReceived = true; // Valid command received

        // ★ ADDED: Update watchdog + reset PID state
        lastCmdMs = millis();
        integral_R = 0.0f;
        prevErr_R = 0.0f;
      }

      bufferIndex = 0; // Reset buffer
    } 
    else {
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
 * @param Kp Proportional gain.
 * @param Ki Integral gain.
 * @param Kd Derivative gain.
 * @return The PID command value.
 */
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
  
  // Configure pins for the right side
  pinMode(M4_ENC_A, INPUT_PULLUP);
  pinMode(M4_ENC_B, INPUT_PULLUP);
  pinMode(M4_PWM, OUTPUT); pinMode(M4_DIR, OUTPUT);
  pinMode(M5_PWM, OUTPUT); pinMode(M5_DIR, OUTPUT);
  pinMode(M6_PWM, OUTPUT); pinMode(M6_DIR, OUTPUT);
  pinMode(start_pin4, OUTPUT); pinMode(start_pin5, OUTPUT);
  pinMode(start_pin6, OUTPUT);

  // Attach interrupt
  attachInterrupt(digitalPinToInterrupt(M4_ENC_A), ISR_Right, RISING);
  
  // Enable right side drivers
  digitalWrite(start_pin4, HIGH); 
  digitalWrite(start_pin5, HIGH);
  digitalWrite(start_pin6, HIGH);

  // Stop motors before starting
  driveMotor(M4_PWM, M4_DIR, 0);
  
  MsTimer2::set(CONTROL_MS, onTimer);
  MsTimer2::start();
}

void loop() {
  // Continuously check for incoming serial commands
  parseSerial();

  // ★ ADDED: Watchdog (timeout) - Stop if no command received or timeout
  if (!velReceived || (millis() - lastCmdMs) > CMD_TIMEOUT_MS) {
    driveMotor(M4_PWM, M4_DIR, 0);
    driveMotor(M5_PWM, M5_DIR, 0);
    driveMotor(M6_PWM, M6_DIR, 0);
    return;
  }
  
  // Execute control loop based on timer period
  if (!flag_Control) return;
  flag_Control = false;

  // Safely read the encoder value using an atomic block
  long cntR;
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
    cntR = encR;
    encR = 0;
  }

  // Convert target velocity (m/s) to wheel RPM
  double targetRPM_R = vel_right_Sub * 60.0 / WHEEL_CIRC;

  // Convert encoder pulses to motor RPM
  double motorRPM_R = (cntR / (double)PULSES_PER_REV) * (60.0 / dt_s);
  
  // Convert motor RPM to wheel RPM
  double wheelRPM_R = motorRPM_R * GEAR_RATIO;

  // Calculate current actual velocity in m/s (for debugging)
  double actual_vel_right = wheelRPM_R * WHEEL_CIRC / 60.0;

  // ★ ADDED: Apply EMA low-pass filter to RPM before using in PID
  wheelRPM_R_filt = RPM_ALPHA * wheelRPM_R + (1.0 - RPM_ALPHA) * wheelRPM_R_filt;

  // Calculate PID control value (using filtered RPM)
  double cmdR = PID(targetRPM_R, wheelRPM_R_filt, integral_R, prevErr_R, KP, KI, KD);

  // Stop motor if target velocity is very low (Deadzone)
  if (fabs(vel_right_Sub) <= 0.05) {
    cmdR = 0;
    // Reset integral term on stop command to prevent overshoot on next start
    integral_R = 0.0f;
  }

  // Limit PWM output value
  cmdR = constrain(cmdR, -250, 250);

  // Apply control signal to all three right side motors
  driveMotor(M4_PWM, M4_DIR,  cmdR);
  driveMotor(M5_PWM, M5_DIR, -cmdR);
  driveMotor(M6_PWM, M6_DIR, -cmdR);

  // === Serial plotter output for debugging ===
  // Serial.print(vel_right_Sub); Serial.print(" ");       // Right target velocity
  // Serial.print(actual_vel_right); Serial.print(" ");     // Right actual velocity (m/s)
  // Serial.println(cmdR);                                  // Control output (PWM approximation)
}