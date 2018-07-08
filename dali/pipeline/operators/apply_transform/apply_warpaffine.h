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

#ifndef DALI_PIPELINE_OPERATORS_APPLY_TRANSFORM_APPLY_WARP_AFFINE_H_
#define DALI_PIPELINE_OPERATORS_APPLY_TRANSFORM_APPLY_WARP_AFFINE_H_

#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/pipeline/operators/operator.h"

namespace dali {

template <typename Backend>
class ApplyWarpAffine : public Operator<Backend> {
 public:
  explicit inline ApplyWarpAffine(const OpSpec &spec) : 
    Operator<Backend>(spec),
    output_w_(spec.GetArgument<int>("output_width")),
    output_h_(spec.GetArgument<int>("output_height")),
    image_layout_(spec.GetArgument<DALITensorLayout>("image_layout")),
    interp_type_(spec.GetArgument<DALIInterpType>("interp_type")) {

    DALI_ENFORCE(image_layout_ == DALI_NCHW ||
                 image_layout_ == DALI_NHWC, 
                 "Unsupported output layout."
                 "Expected NCHW or NHWC.");
    DALI_ENFORCE(output_w_ > 0 && output_h_ > 0);
    if (image_layout_ == DALI_NCHW)
      channel_idx_ = 0;
    else
      channel_idx_ = 2;
    
    GetNativeInterpFlag();
  }

 protected:
  void GetNativeInterpFlag();

  void RunImpl(Workspace<Backend> *ws, const int idx);
  
  // layout of the input and output tensor (NCHW or NHWC)
  DALITensorLayout image_layout_;
  int channel_idx_;

  // interpolation type (NN, Bilinear, Bicubic)
  DALIInterpType interp_type_;
  int interp_flag_;

  // output width and height of the transformed image
  int output_w_;
  int output_h_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_APPLY_TRANSFORM_APPLY_WARP_AFFINE_H_
