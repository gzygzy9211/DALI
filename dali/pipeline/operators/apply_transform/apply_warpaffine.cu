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
#include "dali/util/npp.h"

namespace dali {

template<>
void ApplyWarpaffine<GPUBackend>::GetNativeInterpFlag()
{
  switch(interp_type_) {
    case DALI_INTERP_NN: interp_flag_ = NPPI_INTER_NEAREST; break;
    case DALI_INTERP_CUBIC: interp_flag_ = NPPI_INTER_CUBIC; break;
    case DALI_INTERP_LINEAR: 
    default: interp_flag_ = NPPI_INTER_LINEAR; break;
  }
}

}  // namespace dali