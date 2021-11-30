#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_PERSON_DETECTION_MAIN_FUNCTIONS_H_
#define TENSORFLOW_LITE_MICRO_EXAMPLES_PERSON_DETECTION_MAIN_FUNCTIONS_H_
#include "tensorflow/lite/c/common.h"

#ifdef __cplusplus
extern "C" {
#endif

void person_detect_setup();

void person_detect_exe(uint16_t *frameBuffer, int8_t *rslt);

#ifdef __cplusplus
}
#endif

#endif  // TENSORFLOW_LITE_MICRO_EXAMPLES_PERSON_DETECTION_MAIN_FUNCTIONS_H_
