
#include "person_detect_example.h"
#include "person_detect_model_data.h"

#include "tensorflow/lite/micro/examples/person_detection/detection_responder.h"

#include "tensorflow/lite/micro/examples/person_detection/model_settings.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;

constexpr int kTensorArenaSize = 136 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];
}  // namespace


void person_detect_setup() {
  tflite::InitializeTarget();

  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(g_person_detect_model_data);


//  static tflite::MicroMutableOpResolver<5> micro_op_resolver;
//  micro_op_resolver.AddAveragePool2D();
//  micro_op_resolver.AddConv2D();
//  micro_op_resolver.AddDepthwiseConv2D();
//  micro_op_resolver.AddReshape();
//  micro_op_resolver.AddSoftmax();
  static tflite::AllOpsResolver resolver;

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;


  interpreter->AllocateTensors();

  input = interpreter->input(0);
}

void person_detect_exe(uint16_t *frameBuffer, int8_t *result) {

	for(int i = 0; i < 96; i ++)
	{
		for(int j = 0; j < 96; j ++)
		{
			uint16_t RGB_sample = frameBuffer[(128*16) + 16 + (i*128) + (j)];
			float B = (float)(RGB_sample & 0x1f) / 128.0;
			float G = (float)((RGB_sample >> 5) & 0x3f) / 128.0;
			float R = (float)(RGB_sample >> 11) / 128.0;
			float sum = (R + G + B);

			input->data.int8[i*96 + j] = (int8_t)((sum - 0.5) * 255);
		}
	}

  interpreter->Invoke();

  TfLiteTensor* output = interpreter->output(0);

  *result = output->data.uint8[kPersonIndex];
}
