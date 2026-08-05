#ifndef __SERVO_H
#define __SERVO_H

void Servo_SetAngle(uint8_t channel, int value);
void Servo_SetAllAngles(int val1, int val2, int val3, int val4);
void Servo_Init(void);

#endif
