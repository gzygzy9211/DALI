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

#include "dali/util/npp.h"

#include "dali/error_handling.h"
#include "data/pipeline/data/tensor_list.h"

namespace dali {

static
NppStatus nppiWarpAffine_8u_P1R(const Npp8u *pSrc[1], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                      Npp8u *pDst[1], int nDstStep, NppiRect oDstROI, 
                                      const double aCoeffs[2][3], int eInterpolation) {
  return nppiWarpAffine_8u_C1R(pSrc[0], oSrcSize, nSrcStep, oSrcROI,
                               pDst[0], nDstStep, oDstROI, aCoeffs, eInterpolation);
}

static
NppStatus nppiWarpAffine_32s_P1R(const Npp32s *pSrc[1], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                       Npp32s *pDst[1], int nDstStep, NppiRect oDstROI, 
                                       const double aCoeffs[2][3], int eInterpolation) {
  return nppiWarpAffine_32s_C1R(pSrc[0], oSrcSize, nSrcStep, oSrcROI,
                                pDst[0], nDstStep, oDstROI, aCoeffs, eInterpolation);
}

static
NppStatus nppiWarpAffine_32f_P1R(const Npp32f *pSrc[1], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                       Npp32f *pDst[1], int nDstStep, NppiRect oDstROI, 
                                       const double aCoeffs[2][3], int eInterpolation) {
  return nppiWarpAffine_32f_C1R(pSrc[0], oSrcSize, nSrcStep, oSrcROI,
                                pDst[0], nDstStep, oDstROI, aCoeffs, eInterpolation);
}

static
NppStatus nppiWarpAffine_64f_P1R(const Npp64f *pSrc[1], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                       Npp64f *pDst[1], int nDstStep, NppiRect oDstROI, 
                                       const double aCoeffs[2][3], int eInterpolation) {
  return nppiWarpAffine_64f_C1R(pSrc[0], oSrcSize, nSrcStep, oSrcROI,
                                pDst[0], nDstStep, oDstROI, aCoeffs, eInterpolation);
}


template <DALIDataType dtype> 
const bool NppWarpAffineWrapper<dtype, DALI_NHWC>::supported_channels_[
  NppWarpAffineWrapper<dtype, DALI_NHWC>::MAX_SUPPORTED_CHANNLES
] = {true, false, true, true};

template <DALIDataType dtype>
void NppWarpAffineWrapper<dtype, DALI_NHWC>::call(
  const TensorList<GPUBackend> &input, const std::vector<NppiRect> &input_rois, 
        TensorList<GPUBackend> *output, float* trans_matrix, int interp_flag) {

  for (int i = 0; i < input.ntensor(); i ++) {
    double coeffs[2][3] = {
      {trans_matrix[6 * i + 0], trans_matrix[6 * i + 1], trans_matrix[6 * i + 2]},
      {trans_matrix[6 * i + 3], trans_matrix[6 * i + 4], trans_matrix[6 * i + 5]}
    };
    const npp_t *pSrc = input.tensor<npp_t>(i);
    auto input_shape = input.tensor_shape(i);
    auto output_shape = output->tensor_shape(i);
    npp_t *pDst = output->mutable_tensor(i);

    // input_shape and output_shape are expected to be of *size 3*
    const int in_h = input_shape[0]
    const int in_w = input_shape[1];
    const int channels = input_shape[2];
    const int out_h = output_shape[0];
    const int out_w = output_shape[1];
    DALI_ENFORCE(channels <= MAX_SUPPORTED_CHANNLES && supported_channels_[channels - 1],
                 "Unsupported number of channels for NHWC layout");
    
    warp_affine_func_t func = warp_impl_[channels - 1];
    const NppiSize srcSize = {in_w, in_h};
    const NppiRect dstROI = {0, 0, out_w, out_h};
    const int srcLineStep = sizeof(npp_t) * channels * in_w;
    const int dstLineStep = sizeof(npp_t) * channels * out_w;

    DALI_CHECK_NPP(func(pSrc, srcSize, srcLineStep, input_rois[i], 
                        pDst,          dstLineStep, dstROI, 
                        coeffs, interp_flag));
  }
}

template <DALIDataType dtype>
void NppWarpAffineWrapper<dtype, DALI_NCHW>::call(
  const TensorList<GPUBackend> &input, const std::vector<NppiRect> &input_rois, 
        TensorList<GPUBackend> *output, float* trans_matrix, int interp_flag) {

  for (int i = 0; i < input.ntensor(); i ++) {
    double coeffs[2][3] = {
      {trans_matrix[6 * i + 0], trans_matrix[6 * i + 1], trans_matrix[6 * i + 2]},
      {trans_matrix[6 * i + 3], trans_matrix[6 * i + 4], trans_matrix[6 * i + 5]}
    };
    const npp_t *pSrc = input.tensor<npp_t>(i);
    auto input_shape = input.tensor_shape(i);
    auto output_shape = output->tensor_shape(i);
    npp_t *pDst = output->mutable_tensor(i);

    // input_shape and output_shape are expected to be of *size 3*
    const int channels = input_shape[0];
    const int in_h = input_shape[1]
    const int in_w = input_shape[2];
    const int out_h = output_shape[1];
    const int out_w = output_shape[2];

    const int in_c_step = in_h * in_w;
    const int out_c_step = out_h * out_w;

    const NppiSize srcSize = {in_w, in_h};
    const NppiRect dstROI = {0, 0, out_w, out_h};
    const int srcLineStep = sizeof(npp_t) * in_w;
    const int dstLineStep = sizeof(npp_t) * out_w;

    int c_remain = channels;
    const npp_t *pCurSrc = pSrc;
    npp_t *pCurDst = pDst;
    for (int in_c_stride = 4 * in_c_step, int out_c_stride = 4 * out_c_step;
         c_remain >= 4; 
         c_remain -= 4, pCurSrc += in_c_stride, pCurDst += out_c_stride) {
        const npp_t *pArrSrc[] = {pCurSrc, pCurSrc + in_c_step, 
                                  pCurSrc + 2 * in_c_step, pCurSrc + 3 * in_c_step};
        npp_t *pArrDst[] = {pCurDst, pCurDst + out_c_step, 
                            pCurDst + 2 * out_c_step, pCurDst + 3 * out_c_step};
        DALI_CHECK_NPP(warp_impl_[3](pArrSrc, srcSize, srcLineStep, input_rois[i], 
                                     pArrDst,          dstLineStep, dstROI, 
                                     coeffs, interp_flag));
    }
    for (int in_c_stride = 3 * in_c_step, int out_c_stride = 3 * out_c_step;
         c_remain >= 3; 
         c_remain -= 3, pCurSrc += in_c_stride, pCurDst += out_c_stride) {
        const npp_t *pArrSrc[] = {pCurSrc, pCurSrc + in_c_step, 
                                  pCurSrc + 2 * in_c_step};
        npp_t *pArrDst[] = {pCurDst, pCurDst + out_c_step, 
                            pCurDst + 2 * out_c_step};
        DALI_CHECK_NPP(warp_impl_[2](pArrSrc, srcSize, srcLineStep, input_rois[i], 
                                     pArrDst,          dstLineStep, dstROI, 
                                     coeffs, interp_flag));
    }
    for (int in_c_stride = in_c_step, int out_c_stride = out_c_step;
         c_remain >= 1; 
         c_remain -= 1, pCurSrc += in_c_stride, pCurDst += out_c_stride) {
        const npp_t *pArrSrc[] = {pCurSrc};
        npp_t *pArrDst[] = {pCurDst};
        DALI_CHECK_NPP(warp_impl_[0](pArrSrc, srcSize, srcLineStep, input_rois[i], 
                                     pArrDst,          dstLineStep, dstROI, 
                                     coeffs, interp_flag));
    }
  }
}

template <> 
const typename NppWarpAffineWrapper<DALI_UINT8, DALI_NHWC>::warp_affine_func_t warp_impl_[
  NppWarpAffineWrapper<DALI_UINT8, DALI_NHWC>::MAX_SUPPORTED_CHANNLES
] = {nppiWarpAffine_8u_C1R, nullptr, nppiWarpAffine_8u_C3R, nppiWarpAffine_8u_C4R};

template <> 
const typename NppWarpAffineWrapper<DALI_INT32, DALI_NHWC>::warp_affine_func_t warp_impl_[
  NppWarpAffineWrapper<DALI_INT32, DALI_NHWC>::MAX_SUPPORTED_CHANNLES
] = {nppiWarpAffine_32s_C1R, nullptr, nppiWarpAffine_32s_C3R, nppiWarpAffine_32s_C4R};

template <> 
const typename NppWarpAffineWrapper<DALI_FLOAT, DALI_NHWC>::warp_affine_func_t warp_impl_[
  NppWarpAffineWrapper<DALI_FLOAT, DALI_NHWC>::MAX_SUPPORTED_CHANNLES
] = {nppiWarpAffine_32f_C1R, nullptr, nppiWarpAffine_32f_C3R, nppiWarpAffine_32f_C4R};

template <> 
const typename NppWarpAffineWrapper<DALI_FLOAT64, DALI_NHWC>::warp_affine_func_t warp_impl_[
  NppWarpAffineWrapper<DALI_FLOAT64, DALI_NHWC>::MAX_SUPPORTED_CHANNLES
] = {nppiWarpAffine_64f_C1R, nullptr, nppiWarpAffine_64f_C3R, nppiWarpAffine_64f_C4R};

template <> 
const typename NppWarpAffineWrapper<DALI_UINT8, DALI_NCHW>::warp_affine_func_t warp_impl_[4] = {
  nppiWarpAffine_8u_P1R, nullptr, nppiWarpAffine_8u_P3R, nppiWarpAffine_8u_P4R
};

template <> 
const typename NppWarpAffineWrapper<DALI_INT32, DALI_NCHW>::warp_affine_func_t warp_impl_[4] = {
  nppiWarpAffine_32s_P1R, nullptr, nppiWarpAffine_32s_P3R, nppiWarpAffine_32s_P4R
};

template <> 
const typename NppWarpAffineWrapper<DALI_FLOAT, DALI_NCHW>::warp_affine_func_t warp_impl_[4] = {
  nppiWarpAffine_32f_P1R, nullptr, nppiWarpAffine_32f_P3R, nppiWarpAffine_32f_P4R
};

template <> 
const typename NppWarpAffineWrapper<DALI_FLOAT64, DALI_NCHW>::warp_affine_func_t warp_impl_[4] = {
  nppiWarpAffine_64f_P1R, nullptr, nppiWarpAffine_64f_P3R, nppiWarpAffine_64f_P4R
};


int NPPInterpForDALIInterp(DALIInterpType type, NppiInterpolationMode *npp_type) {
  switch (type) {
  case DALI_INTERP_NN:
    *npp_type =  NPPI_INTER_NN;
    break;
  case DALI_INTERP_LINEAR:
    *npp_type =  NPPI_INTER_LINEAR;
    break;
  case DALI_INTERP_CUBIC:
    *npp_type =  NPPI_INTER_CUBIC;
    break;
  default:
    return DALIError;
  }
  return DALISuccess;
}

}  // namespace dali
