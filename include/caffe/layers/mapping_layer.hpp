#ifndef CAFFE_MAPPING_LAYER_HPP_
#define CAFFE_MAPPING_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Pools the input image by taking the max, average, etc. within regions.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class MappingLayer : public Layer<Dtype> {
 public:
  explicit MappingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Mapping"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  // MAX POOL layers can output an extra top blob for the mask;
  // others can only output the pooled inputs.
  virtual inline int MaxTopBlobs() const {
    return 1;
  }

 protected:
 
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // bool learn;//if learn == false : degrade to traditional dilation
  int kernel;//equal to dilation of previous layer
  Dtype lr;

  Blob<Dtype> mapping_idx_;
  Blob<Dtype> w_lr_xh_;
  Blob<Dtype> w_lr_hh_;
  Blob<Dtype> w_lr_hy_;
  Blob<Dtype> w_rl_xh_;
  Blob<Dtype> w_rl_hh_;
  Blob<Dtype> w_rl_hy_;
  Blob<Dtype> w_ud_xh_;
  Blob<Dtype> w_ud_hh_;
  Blob<Dtype> w_ud_hy_;
  Blob<Dtype> w_du_xh_;
  Blob<Dtype> w_du_hh_;
  Blob<Dtype> w_du_hy_;

  Blob<Dtype> bottom_lr_;
  Blob<Dtype> bottom_rl_;
  Blob<Dtype> bottom_ud_;
  Blob<Dtype> bottom_du_;

  Blob<Dtype> hidden_lr_;
  Blob<Dtype> hidden_rl_;
  Blob<Dtype> hidden_ud_;
  Blob<Dtype> hidden_du_;  
};

}  // namespace caffe

#endif  // CAFFE_MAPPING_LAYER_HPP_
