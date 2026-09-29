//
// Created by twqb on 9/26/26.
//

void encoder_forward(float* out,
                   int* inp, float* wte, float* wpe,
                   int B, int T, int C) {
  // out (B,T,C)，行优先（row-major），out[b][t][i]  =>  out[b*T*C + t*C + i]
  // inp (B,T)，输入 token id 序列，每个元素是词表索引, inp[b][t] => inp[b*T + t]
  // wte (V,C), token embedding 矩阵，V 是词表大小，第 ix 个 token 的向量是 wte[ix*C .. ix*C+C-1]
  // wpe (maxT,C)，position embeddings 矩阵, 第 t 个位置的向量是 wpe[t*C .. t*C+C-1]
  // B 是 批次，T 是 序列长度，C 是 embedding dim
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      // seek to the output position in out[b,t,:]
      float* out_bt = out + b * T * C + t * C;
      // get the index of the token at inp[b, t]
      int ix = inp[b * T + t];
      // seek to the position in wte corresponding to the token
      float* wte_ix = wte + ix * C;
      // seek to the position in wpe corresponding to the position
      float* wpe_t = wpe + t * C;
      // add the two vectors and store the result in out[b,t,:]
      for (int i = 0; i < C; i++) {
        out_bt[i] = wte_ix[i] + wpe_t[i];
      }
    }
  }
}

void encoder_backward(float* dwte, float* dwpe,
                      float* dout, int* inp,
                      int B, int T, int C) {
  // dwte (V, C)，wte 的梯度，dwte[v][i] → dwte[v*C + i]
  // dwpe (maxT, C)，wpe 的梯度，dwpe[s][i] → dwpe[s*C + i]
  // dout (B, T, C)，out 的梯度，dout[b][t][i] → dout[b*T*C + t*C + i]
  // inp (B, T)，token id 序列，inp[b][t] → inp[b*T + t]
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      float* dout_bt = dout + b * T * C + t * C;
      int ix = inp[b * T + t];
      float* dwte_ix = dwte + ix * C;
      float* dwpe_t = dwpe + t * C;
      for (int i = 0; i < C; i++) {
        float d = dout_bt[i];
        dwte_ix[i] += d;
        dwpe_t[i] += d;
      }
    }
  }
}

void layernorm_forward(float* out, float* mean, float* rstd,
                       float* inp, float* weight, float* bias,
                       int B, int T, int C) {
  // reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
  // inp (B, T, C)，token id 序列
  // mean and rstd are (B,T) buffers, to be used later in backward pass
  // at each position (b,t) of the input, the C-dimensional vector
  // of activations gets normalized, then scaled and shifted
  float eps = 1e-5f;
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      // seek to the input position inp[b,t,:]
      float* x = inp + b * T * C + t * C;
      // calculate the mean
      float m = 0.0f;
      for (int i = 0; i < C; i++) {
        m += x[i];
      }
      m = m/C;
      // calculate the variance (without any bias correction)
      float v = 0.0f;
      for (int i = 0; i < C; i++) {
        float xshift = x[i] - m;
        v += xshift * xshift;
      }
      v = v/C;
      // calculate the rstd (reciprocal standard deviation)
      float s = 1.0f / sqrtf(v + eps);
      // seek to the output position in out[b,t,:]
      float* out_bt = out + b * T * C + t * C;
      for (int i = 0; i < C; i++) {
        float n = (s * (x[i] - m)); // normalize
        float o = n * weight[i] + bias[i]; // scale and shift
        out_bt[i] = o; // write
      }
      // cache the mean and rstd for the backward pass later
      mean[b * T + t] = m;
      rstd[b * T + t] = s;
    }
  }
}

typedef struct {
  int max_seq_len; // max sequence length, e.g. 1024
  int vocab_size; // vocab size, e.g. 50257
  int padded_vocab_size; // padded to e.g. %128==0, 50304
  int num_layers; // number of layers, e.g. 12
  int num_heads; // number of heads in attention, e.g. 12
  int channels; // number of channels, e.g. 768
} GPT2Config;

// the parameters of the model
#define NUM_PARAMETER_TENSORS 16
typedef struct {
  float* wte; // (V, C)
  float* wpe; // (maxT, C)
  float* ln1w; // (L, C)
  float* ln1b; // (L, C)
  float* qkvw; // (L, 3*C, C)
  float* qkvb; // (L, 3*C)
  float* attprojw; // (L, C, C)
  float* attprojb; // (L, C)
  float* ln2w; // (L, C)
  float* ln2b; // (L, C)
  float* fcw; // (L, 4*C, C)
  float* fcb; // (L, 4*C)
  float* fcprojw; // (L, C, 4*C)
  float* fcprojb; // (L, C)
  float* lnfw; // (C)
  float* lnfb; // (C)
} ParameterTensors;

void fill_in_parameter_sizes(size_t* param_sizes, GPT2Config config) {
  size_t Vp = config.padded_vocab_size;
  size_t C = config.channels;
  size_t maxT = config.max_seq_len;
  size_t L = config.num_layers;
  param_sizes[0] = Vp * C; // wte
  param_sizes[1] = maxT * C; // wpe
  param_sizes[2] = L * C; // ln1w
  param_sizes[3] = L * C; // ln1b
  param_sizes[4] = L * (3 * C) * C; // qkvw
  param_sizes[5] = L * (3 * C); // qkvb
  param_sizes[6] = L * C * C; // attprojw
  param_sizes[7] = L * C; // attprojb
  param_sizes[8] = L * C; // ln2w
  param_sizes[9] = L * C; // ln2b
  param_sizes[10] = L * (4 * C) * C; // fcw
  param_sizes[11] = L * (4 * C); // fcb
  param_sizes[12] = L * C * (4 * C); // fcprojw
  param_sizes[13] = L * C; // fcprojb
  param_sizes[14] = C; // lnfw
  param_sizes[15] = C; // lnfb
}

typedef struct {
  GPT2Config config;
  // the weights (parameters) of the model, and their sizes
  ParameterTensors params;
  size_t param_sizes[NUM_PARAMETER_TENSORS];
  float* params_memory;
  size_t num_parameters;
  // gradients of the weights
  ParameterTensors grads;
  float* grads_memory;
  // buffers for the AdamW optimizer
  float* m_memory;
  float* v_memory;
  // the activations of the model, and their sizes
  ActivationTensors acts;
  size_t act_sizes[NUM_ACTIVATION_TENSORS];
  float* acts_memory;
  size_t num_activations;
  // gradients of the activations
  ActivationTensors grads_acts;
  float* grads_acts_memory;
  // other run state configuration
  int batch_size; // the batch size (B) of current forward pass
  int seq_len; // the sequence length (T) of current forward pass
  int* inputs; // the input tokens for the current forward pass
  int* targets; // the target tokens for the current forward pass
  float mean_loss; // after a forward pass with targets, will be populated with the mean loss
} GPT2;

#ifndef TESTING
// if we are TESTING (see test_gpt2.c), we'll skip the int main below

// main training loop
int main() {

  // build the GPT-2 model from a checkpoint
  GPT2 model;
  gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");