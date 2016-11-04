#include <algorithm>
#include <vector>

#include "caffe/layers/mapping_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MappingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  MappingParameter param = this->layer_param_.mapping_param();
  // learn = param.learn();
  
  // kernel= param.kernel();
  // lr = Dtype(param.lr());

  lr = Dtype(1e-10);

  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(12);

    vector<int> sz;
    sz.push_back(1);
    for ( int i = 0 ; i < 12; ++i){
      this->blobs_[i].reset(new Blob<Dtype>(sz));
      caffe_set(this->blobs_[i]->count(), Dtype(1.),
                this->blobs_[i]->mutable_cpu_data());
    }
   
    this->blobs_[1]->mutable_cpu_data()[0] = Dtype(0);
    this->blobs_[4]->mutable_cpu_data()[0] = Dtype(0);
    this->blobs_[7]->mutable_cpu_data()[0] = Dtype(0);
    this->blobs_[10]->mutable_cpu_data()[0] = Dtype(0);
  }
}

template <typename Dtype>
void MappingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  top[0]->ReshapeLike(*bottom[0]);

  bottom_lr_.ReshapeLike(*bottom[0]);
  bottom_rl_.ReshapeLike(*bottom[0]);
  bottom_ud_.ReshapeLike(*bottom[0]);
  bottom_du_.ReshapeLike(*bottom[0]);

  hidden_lr_.ReshapeLike(*bottom[0]);
  hidden_rl_.ReshapeLike(*bottom[0]);
  hidden_ud_.ReshapeLike(*bottom[0]);
  hidden_du_.ReshapeLike(*bottom[0]);
  
  vector<int> sz;
  sz.push_back(1);

  w_lr_xh_.Reshape(sz);
  w_lr_hh_.Reshape(sz);
  w_lr_hy_.Reshape(sz);
  w_rl_xh_.Reshape(sz);
  w_rl_hh_.Reshape(sz);
  w_rl_hy_.Reshape(sz);
  w_ud_xh_.Reshape(sz);
  w_ud_hh_.Reshape(sz);
  w_ud_hy_.Reshape(sz);
  w_du_xh_.Reshape(sz);
  w_du_hh_.Reshape(sz);
  w_du_hy_.Reshape(sz);
}

template <typename Dtype>
void MappingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* bottom_lr_data = bottom_lr_.mutable_cpu_data();
  Dtype* bottom_rl_data = bottom_rl_.mutable_cpu_data();
  Dtype* bottom_du_data = bottom_du_.mutable_cpu_data();
  Dtype* bottom_ud_data = bottom_ud_.mutable_cpu_data();
  
  Dtype* hidden_lr_data = hidden_lr_.mutable_cpu_data();
  Dtype* hidden_rl_data = hidden_rl_.mutable_cpu_data();
  Dtype* hidden_du_data = hidden_du_.mutable_cpu_data();
  Dtype* hidden_ud_data = hidden_ud_.mutable_cpu_data();

	caffe_cpu_scale(  w_lr_xh_.count(), Dtype(1), this->blobs_[0]->cpu_data(),   w_lr_xh_.mutable_cpu_data());
	caffe_cpu_scale(  w_lr_hh_.count(), Dtype(1), this->blobs_[1]->cpu_data(),   w_lr_hh_.mutable_cpu_data());
	caffe_cpu_scale(  w_lr_hy_.count(), Dtype(1), this->blobs_[2]->cpu_data(),   w_lr_hy_.mutable_cpu_data());
	caffe_cpu_scale(  w_rl_xh_.count(), Dtype(1), this->blobs_[3]->cpu_data(),   w_rl_xh_.mutable_cpu_data());
	caffe_cpu_scale(  w_rl_hh_.count(), Dtype(1), this->blobs_[4]->cpu_data(),   w_rl_hh_.mutable_cpu_data());
	caffe_cpu_scale(  w_rl_hy_.count(), Dtype(1), this->blobs_[5]->cpu_data(),   w_rl_hy_.mutable_cpu_data());
	caffe_cpu_scale(  w_ud_xh_.count(), Dtype(1), this->blobs_[6]->cpu_data(),   w_ud_xh_.mutable_cpu_data());
	caffe_cpu_scale(  w_ud_hh_.count(), Dtype(1), this->blobs_[7]->cpu_data(),   w_ud_hh_.mutable_cpu_data());
	caffe_cpu_scale(  w_ud_hy_.count(), Dtype(1), this->blobs_[8]->cpu_data(),   w_ud_hy_.mutable_cpu_data());
	caffe_cpu_scale(  w_du_xh_.count(), Dtype(1), this->blobs_[9]->cpu_data(),   w_du_xh_.mutable_cpu_data());
	caffe_cpu_scale(  w_du_hh_.count(), Dtype(1), this->blobs_[10]->cpu_data(),   w_du_hh_.mutable_cpu_data());
	caffe_cpu_scale(  w_du_hy_.count(), Dtype(1), this->blobs_[11]->cpu_data(),   w_du_hy_.mutable_cpu_data());

  //int num = bottom[0]->shape(0);
  int  channels_ = bottom[0]->channels();
  int  height_ = bottom[0]->height();
  int  width_ = bottom[0]->width();

  caffe_set(top[0]->count(), Dtype(0), top_data);
  caffe_set(bottom_lr_.count(), Dtype(0), bottom_lr_data);
  caffe_set(bottom_rl_.count(), Dtype(0), bottom_rl_data);
  caffe_set(bottom_ud_.count(), Dtype(0), bottom_ud_data);
  caffe_set(bottom_du_.count(), Dtype(0), bottom_du_data);
  caffe_set(hidden_lr_.count(), Dtype(0), hidden_lr_data);
  caffe_set(hidden_rl_.count(), Dtype(0), hidden_rl_data);
  caffe_set(hidden_ud_.count(), Dtype(0), hidden_ud_data);
  caffe_set(hidden_du_.count(), Dtype(0), hidden_du_data);


  for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < height_; ++ph) {
          int index = width_ * ph; 
          hidden_lr_data[index] = std::max(w_lr_hh_.cpu_data()[0] * 0 + w_lr_xh_.cpu_data()[0]* bottom_data[index] , Dtype(0));
          bottom_lr_data[index] = std::max(hidden_lr_data[index] * w_lr_hy_.cpu_data()[0], Dtype(0));
          top_data[index] += bottom_lr_data[index]; 

          index = width_ * ph + width_ - 1;
          hidden_rl_data[ index ] = std::max(w_rl_hh_.cpu_data()[0] * 0 + w_rl_xh_.cpu_data()[0]* bottom_data[index] , Dtype(0));
          bottom_rl_data[index] = std::max(hidden_rl_data[index] * w_rl_hy_.cpu_data()[0], Dtype(0));
          top_data[index] += bottom_rl_data[index];

          for (int pw = 1; pw < width_; ++pw) {
              index = width_ * ph + pw;
              hidden_lr_data[index] = std::max(w_lr_hh_.cpu_data()[0] * hidden_lr_data[index -1] + w_lr_xh_.cpu_data()[0]* bottom_data[index] , Dtype(0));
              bottom_lr_data[index] = std::max(hidden_lr_data[index] * w_lr_hy_.cpu_data()[0] , Dtype(0));
              top_data[index] += bottom_lr_data[index]; 
              
              index = width_ * ph + (width_ -1 ) - pw;
              hidden_rl_data[index] = std::max(w_rl_hh_.cpu_data()[0] * hidden_rl_data[index +1] + w_rl_xh_.cpu_data()[0]* bottom_data[index] , Dtype(0));
              bottom_rl_data[index] = std::max(hidden_rl_data[index] * w_rl_hy_.cpu_data()[0] , Dtype(0));
              top_data[index] += bottom_rl_data[index];
            }
        }

        for (int pw = 0; pw < width_; ++pw) {
          int index = pw;
          hidden_ud_data[pw] = std::max(w_ud_hh_.cpu_data()[0] * 0 + w_ud_xh_.cpu_data()[0]* bottom_data[pw] , Dtype(0));
          bottom_ud_data[pw] = std::max(hidden_ud_data[pw] * w_ud_hy_.cpu_data()[0], Dtype(0)); 
          top_data[pw] += bottom_ud_data[pw]; 
          index = height_ * width_ - width_  + pw;
          hidden_du_data[ index ] = std::max(w_du_hh_.cpu_data()[0] * 0 + w_du_xh_.cpu_data()[0]* bottom_data[index] , Dtype(0));
          bottom_du_data[index] = std::max(hidden_du_data[index] * w_du_hy_.cpu_data()[0], Dtype(0));
          top_data[index] += bottom_du_data[index];

          for (int ph = 1; ph < height_; ++ph) {
              int index = width_ * ph + pw;
              hidden_ud_data[index] = std::max(w_ud_hh_.cpu_data()[0] * hidden_ud_data[index - width_] + w_ud_xh_.cpu_data()[0]* bottom_data[index] , Dtype(0));
              bottom_ud_data[index] = std::max(hidden_ud_data[index] * w_ud_hy_.cpu_data()[0] , Dtype(0));
              top_data[index] += bottom_ud_data[index]; 

              index = width_ * (height_ - ph -1) + pw;
              hidden_du_data[index] = std::max(w_du_hh_.cpu_data()[0] * hidden_du_data[index + width_] + w_du_xh_.cpu_data()[0]* bottom_data[index] , Dtype(0));
              bottom_du_data[index] = std::max(hidden_du_data[index] * w_du_hy_.cpu_data()[0] , Dtype(0));
              top_data[index] += bottom_du_data[index];
            }
        }
        
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
        bottom_lr_data += bottom_lr_.offset(0, 1);
        bottom_rl_data += bottom_rl_.offset(0, 1);
        bottom_ud_data += bottom_du_.offset(0, 1);
        bottom_du_data += bottom_ud_.offset(0, 1);
        hidden_lr_data += hidden_lr_.offset(0, 1);
        hidden_rl_data += hidden_rl_.offset(0, 1);
        hidden_ud_data += hidden_ud_.offset(0, 1);
        hidden_du_data += hidden_du_.offset(0, 1);
      }
    }
    top_data = top[0]->mutable_cpu_data();
    const int count  = top[0]->count();
    for ( int i =0 ; i < count; ++i)
      top_data[i] /= 4 ; 
}

template <typename Dtype>
void MappingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  Dtype* bottom_lr_diff = bottom_lr_.mutable_cpu_diff();
  Dtype* bottom_rl_diff = bottom_rl_.mutable_cpu_diff();
  Dtype* bottom_du_diff = bottom_du_.mutable_cpu_diff();
  Dtype* bottom_ud_diff = bottom_ud_.mutable_cpu_diff();
  
  Dtype* hidden_lr_diff = hidden_lr_.mutable_cpu_diff();
  Dtype* hidden_rl_diff = hidden_rl_.mutable_cpu_diff();
  Dtype* hidden_du_diff = hidden_du_.mutable_cpu_diff();
  Dtype* hidden_ud_diff = hidden_ud_.mutable_cpu_diff();

  const Dtype* bottom_lr_data = bottom_lr_.cpu_data();
  const Dtype* bottom_rl_data = bottom_rl_.cpu_data();
  const Dtype* bottom_du_data = bottom_du_.cpu_data();
  const Dtype* bottom_ud_data = bottom_ud_.cpu_data();
  
  const Dtype* hidden_lr_data = hidden_lr_.cpu_data();
  const Dtype* hidden_rl_data = hidden_rl_.cpu_data();
  const Dtype* hidden_du_data = hidden_du_.cpu_data();
  const Dtype* hidden_ud_data = hidden_ud_.cpu_data();

  Dtype* wv_lr_xh = w_lr_xh_.mutable_cpu_data();
  Dtype* wv_lr_hh = w_lr_hh_.mutable_cpu_data();
  Dtype* wv_lr_hy = w_lr_hy_.mutable_cpu_data();
  Dtype* wv_rl_xh = w_rl_xh_.mutable_cpu_data();
  Dtype* wv_rl_hh = w_rl_hh_.mutable_cpu_data();
  Dtype* wv_rl_hy = w_rl_hy_.mutable_cpu_data();
  Dtype* wv_ud_xh = w_ud_xh_.mutable_cpu_data();
  Dtype* wv_ud_hh = w_ud_hh_.mutable_cpu_data();
  Dtype* wv_ud_hy = w_ud_hy_.mutable_cpu_data();
  Dtype* wv_du_xh = w_du_xh_.mutable_cpu_data();
  Dtype* wv_du_hh = w_du_hh_.mutable_cpu_data();
  Dtype* wv_du_hy = w_du_hy_.mutable_cpu_data();


  int num_ = bottom[0]->shape(0);
  int  channels_ = bottom[0]->channels();
  int  height_ = bottom[0]->height();
  int  width_ = bottom[0]->width();

  int count = top[0]->count();
  CHECK_EQ(bottom_lr_.count(), count);
  for (int i = 0; i< count; ++i){
    bottom_lr_diff[i] = top_diff[i] * (bottom_lr_data[i] > 0) /4. * wv_lr_hy[0];
    bottom_rl_diff[i] = top_diff[i] * (bottom_rl_data[i] > 0) /4. * wv_rl_hy[0] ;
    bottom_ud_diff[i] = top_diff[i] * (bottom_ud_data[i] > 0) /4. * wv_ud_hy[0] ;
    bottom_du_diff[i] = top_diff[i] * (bottom_du_data[i] > 0) /4. * wv_du_hy[0] ;

  }
  for ( int i = 0 ; i < count; ++i){
    wv_lr_hy[0] -= lr * bottom_lr_data[i] * top_diff[i];
    wv_rl_hy[0] -= lr * bottom_rl_data[i] * top_diff[i];
    wv_ud_hy[0] -= lr * bottom_ud_data[i] * top_diff[i];
    wv_du_hy[0] -= lr * bottom_du_data[i] * top_diff[i];
  }
  LOG(INFO) << "bottom bp correct!";
  int base = 0;
  for (int n = 0; n < num_; ++n) {
      for (int c = 0; c < channels_; ++c) {
        int index = 0;
        for (int ph = 0; ph < height_; ++ph) {
          index = base + ph * width_ + width_ - 1;
          hidden_lr_diff[index] = bottom_lr_diff[index] ;
          index = base + ph * width_;
          hidden_rl_diff[index] = bottom_rl_diff[index] ;
          for (int pw = width_ -1; pw > 0; --pw){
            index = base + ph * width_ + pw;
            hidden_lr_diff[index -1] = bottom_lr_diff[index-1] + bottom_lr_diff[index] * ((hidden_lr_data[index] > 0)) * wv_lr_hh[0] ;
            index = base + ph * width_ + width_ - pw;
            hidden_rl_diff[index ] = bottom_rl_diff[index] + bottom_rl_diff[index -1] * ((hidden_rl_data[index-1] > 0)) * wv_rl_hh[0] ;
          }
        }
        for ( int pw = 0; pw < width_; ++pw){
          index = base  + pw;
          hidden_du_diff[index] = bottom_du_diff[index];
          index = base + height_ * width_ - width_ + pw;
          hidden_ud_diff[index] = bottom_ud_diff[index];
          for (int ph = height_ -1 ; ph > 0; --ph){
            index = base + ph * width_ - width_ + pw;
            hidden_ud_diff[index] = bottom_ud_diff[index] + bottom_ud_diff[index + width_] * (hidden_ud_data[index+width_] > 0) * wv_ud_hh[0] ;
            index = base + (height_ - ph) + pw;
            hidden_du_diff[index] = bottom_du_diff[index] + bottom_du_diff[index - width_] * (hidden_ud_data[index + width_] > 0) * wv_du_hh[0] ;  
          }
        }
        base += height_ * width_;

      }
    }
    LOG(INFO) << "hidden bp correct!";
  int base_w = 0;
  for (int n = 0; n < num_; ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < height_; ++ph) {
          for (int pw = width_ - 2; pw >= 0 ; --pw){
            int index = base_w +  ph * width_ + pw;
            wv_lr_hh[0] -= lr * hidden_lr_data[index] * bottom_lr_diff[index+1];
            index = base_w + ph * width_ + width_ - 1 - pw;
            wv_rl_hh[0] -= lr * hidden_rl_data[index] * bottom_rl_diff[index-1]; 
          }
        }
        for ( int pw = 0; pw < width_; ++pw){
          for (int ph = height_ - 2 ; ph >= 0; --ph){
            int index = base_w + ph * width_ + pw;
            wv_ud_hh[0] -= lr * hidden_ud_data[index] * bottom_ud_diff[index + width_];
            index = base_w + (height_ - ph -1) * width_ + pw;
            wv_du_hh[0] -= lr * hidden_du_data[index] * bottom_du_diff[index - width_];
          }
        }
      }
    }
    LOG(INFO) << "w_hh update correct!";
  for (int i = 0; i <  count; i ++){
    bottom_diff[i] = hidden_lr_diff[i] * wv_lr_xh[0] + hidden_rl_diff[i] * wv_rl_xh[0] + hidden_du_diff[i] * wv_du_xh[0] + hidden_ud_diff[i] * wv_ud_xh[0] ;
  }
  for ( int i = 0; i<  count; ++i){
    wv_lr_xh[0] -= lr * hidden_lr_diff[i] * bottom_data[i];
    wv_rl_xh[0] -= lr * hidden_rl_diff[i] * bottom_data[i];
    wv_ud_xh[0] -= lr * hidden_ud_diff[i] * bottom_data[i];
    wv_du_xh[0] -= lr * hidden_du_diff[i] * bottom_data[i];
  }
  LOG(INFO) << "w_xh update correct!";
  caffe_cpu_scale(w_lr_xh_.count(),Dtype(1), w_lr_xh_.cpu_data(),   this->blobs_[0]->mutable_cpu_data());
  caffe_cpu_scale(w_lr_hh_.count(),Dtype(1), w_lr_hh_.cpu_data(),   this->blobs_[1]->mutable_cpu_data());
  caffe_cpu_scale(w_lr_hy_.count(),Dtype(1), w_lr_hy_.cpu_data(),   this->blobs_[2]->mutable_cpu_data());
  caffe_cpu_scale(w_rl_xh_.count(),Dtype(1), w_rl_xh_.cpu_data(),   this->blobs_[3]->mutable_cpu_data());
  caffe_cpu_scale(w_rl_hh_.count(),Dtype(1), w_rl_hh_.cpu_data(),   this->blobs_[4]->mutable_cpu_data());
  caffe_cpu_scale(w_rl_hy_.count(),Dtype(1), w_rl_hy_.cpu_data(),   this->blobs_[5]->mutable_cpu_data());
  caffe_cpu_scale(w_ud_xh_.count(),Dtype(1), w_ud_xh_.cpu_data(),   this->blobs_[6]->mutable_cpu_data());
  caffe_cpu_scale(w_ud_hh_.count(),Dtype(1), w_ud_hh_.cpu_data(),   this->blobs_[7]->mutable_cpu_data());
  caffe_cpu_scale(w_ud_hy_.count(),Dtype(1), w_ud_hy_.cpu_data(),   this->blobs_[8]->mutable_cpu_data());
  caffe_cpu_scale(w_du_xh_.count(),Dtype(1), w_du_xh_.cpu_data(),   this->blobs_[9]->mutable_cpu_data());
  caffe_cpu_scale(w_du_hh_.count(),Dtype(1), w_du_hh_.cpu_data(),   this->blobs_[10]->mutable_cpu_data());
  caffe_cpu_scale(w_du_hy_.count(),Dtype(1), w_du_hy_.cpu_data(),   this->blobs_[11]->mutable_cpu_data());

}


#ifdef CPU_ONLY
STUB_GPU(MappingLayer);
#endif

INSTANTIATE_CLASS(MappingLayer);
REGISTER_LAYER_CLASS(Mapping);
}  // namespace caffe
