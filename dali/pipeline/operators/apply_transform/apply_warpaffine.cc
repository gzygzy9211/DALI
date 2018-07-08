// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "dali/pipeline/operators/apply_transform/apply_warpaffine.h"
#include "dali/util/ocv.h"

namespace dali {

inline int GetOcvMatType(DALIDataType dtype, int channels)
{
  int mat_tpye;
  switch(input.type().id()) {
    case DALI_UINT8: mat_tpye = CV_8UC(channels); break;
    case DALI_INT16: mat_type = CV_16SC(channels); break;
    case DALI_INT32: mat_type = CV_32SC(channels); break;
    case DALI_FLOAT: mat_type = CV_32FC(channels); break;
    case DALI_FLOAT64: mat_type = CV_64FC(channels); break;
    default:
      DALI_FAIL(to_string(dtype) + ": Unsupported dtype of OpenCV");
  }
}

template<>
void ApplyWarpAffine<CPUBackend>::GetNativeInterpFlag()
{
  switch(interp_type_) {
    case DALI_INTERP_NN: interp_flag_ = CV_INTER_NEAREST; break;
    case DALI_INTERP_CUBIC: interp_flag_ = CV_INTER_CUBIC; break;
    case DALI_INTERP_LINEAR: 
    default: interp_flag_ = CV_INTER_LINEAR; break;
  }
}

template<>
void ApplyWarpAffine<CPUBackend>::RunImpl(SampleWorkspace * ws, const int idx) {
  auto &image_tensor = ws->Input<CPUBackend>(0);
  auto &trans_tensor = ws->Input<CPUBackend>(1);

  auto output = ws->Output<CPUBackend>(0);

  int mat_type;
  DALI_ENFORCE(IsType<float>(trans_tensor.type()), 
        "Transform matrix must be type of float");
  DALI_ENFORCE(trans_tensor.size() == 6, 
        "warpAffine accept only 2x3 transform matrix");
  DALI_ENFORCE(image_tensor.ndim() == 3,
        "Expects 3-dim image input.");
  cv::Mat trans(2, 3, CV_32F, const_cast<void*>trans_tensor.raw_data());
  const int channels = image_tensor.dim(channel_idx_);
  
  if (image_layout_ == DALI_NHWC)
  {
    const ocv_mat_type = GetOcvMatType(image_tensor.type().id(), channels);
    DALI_ENFORCE(channels < 512, "OpenCV Only Support Channels LESS than 512");
    cv::Mat input(image_tensor.dim(0), 
                  image_tensor.dim(1),
                  ocv_mat_type,
                  const_cast<void*>(image_tensor.raw_data()));
    output->Resize({output_h_, output_w_, channels});
    output->set_type(image_tensor->type());
    cv::Mat output(output_h_, output_w_, ocv_mat_type, output->raw_mutable_data());
    cv::warpAffine(input, output, trans, cv::Size(output_w_, output_h_), interp_flag_);
  }
  else
  {
    const ocv_mat_type = GetOcvMatType(image_tensor.type().id(), 1);
    const input_channel_stride_in_bytes 
      = image_tesnsor.dim(1) * image_tensor.dim(2) * image_tensor.type().size();
    const output_channel_stride_in_bytes
      = output_h_ * output_w_ * image_tensor.type().size();
    
    output->Resize({channels, output_h_, output_w_});
    output->set_type(image_tensor->type());
    
    for (int c = 0, in_offset = 0, out_offset = 0; 
         c < channels; c++, 
         in_offset += input_channel_stride_in_bytes,
         out_offset += output_channel_stride_in_bytes)
    {
      cv::Mat input(image_tensor.dim(1),
                    image_tensor.dim(2),
                    ocv_mat_type,
                    const_cast<void*>(
                      image_tensor.raw_data() + in_offset
                    ));
      cv::Mat output(output_h_, output_w_, ocv_mat_type, 
        output->raw_mutable_data() + out_offset);
      cv::warpAffine(input, output, trans, cv::Size(output_w_, output_h_), interp_flag_);
    }
  }
}

DALI_REGISTER_OPERATOR(ApplyWarpAffine, ApplyWarpAffine<CPUBackend>, CPU);

DALI_SCHEMA(ApplyWarpaffine)
  .DocStr(R"code(Take the input image and the input 2x3 matrix, perform
          warpAffine on the input image according to the 2x3 matrix. After
          the transformation, rectangle region [0, 0, output_w, output_h]
          is cropped out as the output image, just like `warpAffine` in 
          OpenCV)code")
  .NumInput(2)
  .NumOutput(1)
  .AddArg("output_width", 
    R"code(`int`
    Width of the output images)code")
  .AddArg("output_height",
    R"code(`int`
    Height of the output images)code")
  .AddOptionalArg("image_layout",
    R"code(`dali.types.DALITensorLayout`
    Layout of the input images)code", DALI_NHWC)
  .AddOptionalArg("interp_type",
    R"code(`dali.types.DALIInterpType`
    Interpolation type while performing
    the transformation)code", DALI_INTERP_NN);

}  // namespace dali
