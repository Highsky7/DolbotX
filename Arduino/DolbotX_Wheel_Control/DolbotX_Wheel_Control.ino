/**
 * @file DolbotX_Wheel_Control.ino
 * @brief Main control sketch for the DolbotX robot's 6-wheel differential drive system.
 *
 * This sketch runs on an Arduino Mega and is responsible for:
 * - Receiving target wheel velocities (left and right) via serial communication.
 * - Reading motor encoders to measure the actual speed of the two main motors (M1 and M4).
 * - Implementing a PID control loop to match the actual speed to the target speed.
 * - Driving all six motors based on the PID output. The motors on each side
 *   (left: M1, M2, M3; right: M4, M5, M6) are driven with the same command signal,
 *   with directions adjusted accordingly.
 *
 * Serial Command Format: "VL <left_velocity_mps> <right_velocity_mps>\n"
 * Example: "VL 0.5 -0.5\n"
 */

#include <MsTimer2.h>
#include <math.h>

// === Pin Definitions ===
// --- Left Side Motors ---
#define M1_PWM 2      // PWM pin for Motor 1 (with encoder)
#define M1_DIR 3      // Direction pin for Motor 1
#define M1_ENC_A 18   // Encoder A channel for Motor 1
#define M1_ENC_B 19   // Encoder B channel for Motor 1
#define M2_PWM 12     // PWM pin for Motor 2
#define M2_DIR 13     // Direction pin for Motor 2
#define M3_PWM 8      // PWM pin for Motor 3
#define M3_DIR 9      // Direction pin for Motor 3

// --- Right Side Motors ---
#define M4_PWM 4      // PWM pin for Motor 4 (with encoder)
#define M4_DIR 5      // Direction pin for Motor 4
#define M4_ENC_A 20   // Encoder A channel for Motor 4
#define M4_ENC_B 21   // Encoder B channel for Motor 4
#define M5_PWM 11     // PWM pin for Motor 5
#define M5_DIR 10     // Direction pin for Motor 5
#define M6_PWM 6      // PWM pin for Motor 6
#define M6_DIR 7      // Direction pin for Motor 6

// --- Motor Driver Enable Pins ---
#define start_pin1 22
#define start_pin2 24
#define start_pin3 26
#define start_pin4 34
#define start_pin5 30
#define start_pin6 32

// === Constants ===
#define PULSES_PER_REV 350      // Encoder pulses per motor revolution
#define CONTROL_MS 20           // Control loop period in milliseconds
const double dt_s = CONTROL_MS / 1000.0; // Control loop period in seconds
const double GEAR_RATIO = 1.0 / 71.0;    // Gear ratio (motor -> wheel)
const double WHEEL_DIAM = 0.135;         // Wheel diameter [m]
const double WHEEL_CIRC = M_PI * WHEEL_DIAM; // Wheel circumference [m]

// PID Gains
const float KP = 0.8;
const float KI = 0.0;
const float KD = 0.0;
const double I_MAX = 3000.0; // Maximum value for the integral term (anti-windup)

// === Global Variables ===
volatile long encL = 0, encR = 0; // Left and right encoder pulse counts
float vel_left_Sub = 0.0, vel_right_Sub = 0.0; // Target velocities from serial [m/s]
float integral1 = 0.0f, prevErr1 = 0.0f; // PID state variables for left side
float integral4 = 0.0f, prevErr4 = 0.0f; // PID state variables for right side
volatile bool flag_Control = false; // Flag set by timer interrupt to run the control loop
volatile bool velReceived = false;  // Flag to indicate if a valid velocity command has been received

// Buffer for non-blocking serial reception
const int SERIAL_BUFFER_SIZE = 64;
char serialBuffer[SERIAL_BUFFER_SIZE];
int bufferIndex = 0;

/**
 * @brief Interrupt Service Routine for the left encoder.
 * Increments or decrements the left encoder count based on direction.
 */
void ISR_Left() {
  if (digitalRead(M1_ENC_B) == LOW) encL++;
  else encL--;
}

/**
 * @brief Interrupt Service Routine for the right encoder.
 * Increments or decrements the right encoder count based on direction.
 */
void ISR_Right() {
  if (digitalRead(M4_ENC_B) == LOW) encR++;
  else encR--;
}

/**
 * @brief Timer interrupt callback. Sets a flag to trigger the main control loop.
 */
void onTimer() {
  flag_Control = true;
}

/**
 * @brief Robustly parses incoming serial data.
 * Reads characters until a newline is found, then parses the line for
 * velocity commands using strtok and atof.
 * Expected format: "VL <left_vel> <right_vel>\n"
 */
void parseSerial() {
  while (Serial.available() > 0) {
    char incomingChar = Serial.read();

    if (incomingChar == '\n') {
      serialBuffer[bufferIndex] = '\0'; 

      if (strncmp(serialBuffer, "VL", 2) == 0) {
        // Create a copy of the buffer because strtok modifies it.
        char tempBuffer[SERIAL_BUFFER_SIZE];
        strcpy(tempBuffer, serialBuffer);

        char *ptr = strtok(tempBuffer, " "); // Separate the first token ("VL")
        if (ptr != NULL) {
          ptr = strtok(NULL, " "); // Separate the second token (left velocity)
          if (ptr != NULL) {
            vel_left_Sub = atof(ptr); // atof: convert string to float
            
            ptr = strtok(NULL, " "); // Separate the third token (right velocity)
            if (ptr != NULL) {
              vel_right_Sub = atof(ptr);
              velReceived = true; // A valid command was received
            }
          }
        }
      }
      bufferIndex = 0; // Reset buffer for the next line
    } 
    else {
      if (bufferIndex < SERIAL_BUFFER_SIZE - 1) {
        serialBuffer[bufferIndex++] = incomingChar;
      } else {
        // Buffer overflow, reset
        bufferIndex = 0;
      }
    }
  }
}

/**
 * @brief Drives a single motor with a given command value.
 * @param pwmPin The PWM pin for the motor.
 * @param dirPin The direction pin for the motor.
 * @param cmd The command value. Its magnitude is mapped to PWM, and its sign determines direction.
 */
void driveMotor(int pwmPin, int dirPin, double cmd) {
  int pwm = (int)round(fabs(cmd));
  if (pwm > 200) pwm = 200; // Cap PWM value
  if (pwm < 15) pwm = 0;   // Deadzone for motor startup
  digitalWrite(dirPin, (cmd >= 0) ? HIGH : LOW);
  analogWrite(pwmPin, pwm);
}

/**
 * @brief Calculates the PID control output.
 * @param target The target setpoint (e.g., target RPM).
 * @param current The current measured value (e.g., current RPM).
 * @param integral Reference to the integral term for this PID loop.
 * @param prevErr Reference to the previous error term for this PID loop.
 * @return The calculated PID command value.
 */
double PID(double target, double current, float& integral, float& prevErr) {
  double error = target - current;
  integral += error * dt_s;
  integral = constrain(integral, -I_MAX, I_MAX); // Anti-windup
  double derivative = (error - prevErr) / dt_s;
  prevErr = error;
  return KP * error + KI * integral + KD * derivative;
}

/**
 * @brief Setup function, runs once at startup.
 * Initializes serial communication, pin modes, interrupts, and the control timer.
 */
void setup() {
  Serial.begin(57600);

  // Set pin modes
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
  
  // Attach interrupts for encoders
  attachInterrupt(digitalPinToInterrupt(M1_ENC_A), ISR_Left, RISING);
  attachInterrupt(digitalPinToInterrupt(M4_ENC_A), ISR_Right, RISING);

  // Enable all motor drivers
  digitalWrite(start_pin1, HIGH); digitalWrite(start_pin2, HIGH);
  digitalWrite(start_pin3, HIGH); digitalWrite(start_pin4, HIGH);
  digitalWrite(start_pin5, HIGH); digitalWrite(start_pin6, HIGH);

  // Ensure motors are stopped initially
  driveMotor(M1_PWM, M1_DIR, 0);
  driveMotor(M4_PWM, M4_DIR, 0);
  
  // Set up and start the control loop timer
  MsTimer2::set(CONTROL_MS, onTimer);
  MsTimer2::start();
}

/**
 * @brief Main loop.
 * Continuously parses serial data. When the control timer flag is set, it
 * executes the PID control logic.
 */
void loop() {
  parseSerial();

  // If no valid velocity command has been received yet, keep motors stopped.
  if (!velReceived) {
    driveMotor(M1_PWM, M1_DIR, 0); driveMotor(M2_PWM, M2_DIR, 0);
    driveMotor(M3_PWM, M3_DIR, 0); driveMotor(M4_PWM, M4_DIR, 0);
    driveMotor(M5_PWM, M5_DIR, 0); driveMotor(M6_PWM, M6_DIR, 0);
    return;
  }
  
  // Wait for the timer interrupt to set the flag
  if (!flag_Control) return;
  flag_Control = false; // Reset the flag

  // Atomically read and reset encoder counts
  noInterrupts();
  long cntL = encL, cntR = encR;
  encL = encR = 0;
  interrupts();

  // Convert target velocity (m/s) to target wheel RPM
  double targetRPM_L = vel_left_Sub * 60.0 / WHEEL_CIRC;
  double targetRPM_R = vel_right_Sub * 60.0 / WHEEL_CIRC;

  // Calculate current motor RPM from encoder counts
  double motorRPM_L = (cntL / (double)PULSES_PER_REV) * (60.0 / dt_s);
  double motorRPM_R = (cntR / (double)PULSES_PER_REV) * (60.0 / dt_s);

  // Calculate current wheel RPM using the gear ratio
  double wheelRPM_L = motorRPM_L * GEAR_RATIO;
  double wheelRPM_R = motorRPM_R * GEAR_RATIO;

  // Calculate PID command for each side
  double cmdL = PID(targetRPM_L, wheelRPM_L, integral1, prevErr1);
  double cmdR = PID(targetRPM_R, wheelRPM_R, integral4, prevErr4);

  // If target velocity is very low, command a full stop
  if (fabs(vel_left_Sub) <= 0.05) cmdL = 0;
  if (fabs(vel_right_Sub) <= 0.05) cmdR = 0;

  // Constrain command values to PWM limits
  cmdL = constrain(cmdL, -200, 200);
  cmdR = constrain(cmdR, -200, 200);

  // Drive all motors on the left side
  driveMotor(M1_PWM, M1_DIR,  cmdL);   // Main motor (with encoder)
  driveMotor(M2_PWM, M2_DIR, -cmdL);   // Opposite direction to M1
  driveMotor(M3_PWM, M3_DIR, -cmdL);   // Opposite direction to M1

  // Drive all motors on the right side
  driveMotor(M4_PWM, M4_DIR,  cmdR);   // Main motor (with encoder)
  driveMotor(M5_PWM, M5_DIR, -cmdR);   // Opposite direction to M4
  driveMotor(M6_PWM, M6_DIR, -cmdR);   // Opposite direction to M4
}