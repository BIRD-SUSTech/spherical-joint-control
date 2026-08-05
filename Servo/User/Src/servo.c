#include "main.h"
#include "stm32f1xx_hal_tim.h"
#include <stdint.h>

extern TIM_HandleTypeDef htim2;

// the val ranges from -135 to 135 according to the angle that servo can turn

uint16_t int_to_pulse(int val) {

  if (val < -135)
    val = -135;
  if (val > 135)
    val = 135;

  // 线性映射，简单的数学关系
  return (uint16_t)(1500 + (val * 200 / 27));
}

void Servo_SetAngle(uint8_t channel, int value) {

  uint16_t pulse = int_to_pulse(value);

  switch (channel) {
  case 1:
    __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_1, pulse);
    break;
  case 2:
    __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_2, pulse);
    break;
  case 3:
    __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_3, pulse);
    break;
  case 4:
    __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_4, pulse);
    break;
  default:
    break;
  }
}

void Servo_SetAllAngles(int val1, int val2, int val3, int val4) {

  __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_1, int_to_pulse(val1));
  __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_2, int_to_pulse(val2));
  __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_3, int_to_pulse(val3));
  __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_4, int_to_pulse(val4));
}

void Servo_Init(void) {
  HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_1); // PA0
  HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_2); // PA1
  HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_3); // PA2
  HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_4); // PA3

  Servo_SetAllAngles(0, 0, 0, 0);
}
