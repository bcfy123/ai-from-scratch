//
// Created by twqb on 9/26/26.
//

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#ifdef OMP
#include <omp.h>
#endif
// our own utilities
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#include "utils.h"
// defines: dataloader_init, dataloader_reset, dataloader_next_batch, dataloader_free
#include "dataloader.h"

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


// allocate memory for the parameters and point the individual tensors to the right places
float* malloc_and_point_parameters(ParameterTensors* params, size_t* param_sizes) {
  size_t num_parameters = 0;
  for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
    num_parameters += param_sizes[i];
  }
  // malloc all parameters all at once
  float* params_memory = (float*)mallocCheck(num_parameters * sizeof(float));
  // assign all the tensors
  float** ptrs[] = {
    &params->wte, &params->wpe, &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
    &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw, &params->fcb,
    &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb
};
  float* params_memory_iterator = params_memory;
  for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
    *(ptrs[i]) = params_memory_iterator;
    params_memory_iterator += param_sizes[i];
  }
  return params_memory;
}


#define NUM_ACTIVATION_TENSORS 23
typedef struct {
  float* encoded; // (B, T, C)
  float* ln1; // (L, B, T, C)
  float* ln1_mean; // (L, B, T)
  float* ln1_rstd; // (L, B, T)
  float* qkv; // (L, B, T, 3*C)
  float* atty; // (L, B, T, C)
  float* preatt; // (L, B, NH, T, T)
  float* att; // (L, B, NH, T, T)
  float* attproj; // (L, B, T, C)
  float* residual2; // (L, B, T, C)
  float* ln2; // (L, B, T, C)
  float* ln2_mean; // (L, B, T)
  float* ln2_rstd; // (L, B, T)
  float* fch; // (L, B, T, 4*C)
  float* fch_gelu; // (L, B, T, 4*C)
  float* fcproj; // (L, B, T, C)
  float* residual3; // (L, B, T, C)
  float* lnf; // (B, T, C)
  float* lnf_mean; // (B, T)
  float* lnf_rstd; // (B, T)
  float* logits; // (B, T, V)
  float* probs; // (B, T, V)
  float* losses; // (B, T)
} ActivationTensors;

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

void gpt2_build_from_checkpoint(GPT2 *model, const char* checkpoint_path) {

    // read in model from a checkpoint file
    FILE *model_file = fopenCheck(checkpoint_path, "rb");
    int model_header[256];
    freadCheck(model_header, sizeof(int), 256, model_file);
    if (model_header[0] != 20240326) { printf("Bad magic model file\n"); exit(1); }
    if (model_header[1] != 3) {
        printf("Bad version in model file\n");
        printf("---> HINT: try to re-run `python train_gpt2.py`\n");
        exit(1);
    }

    // read in hyperparameters
    size_t maxT, V, Vp, L, NH, C; // size_t to prevent int overflow
    model->config.max_seq_len = maxT = model_header[2];
    model->config.vocab_size = V = model_header[3];
    model->config.num_layers = L = model_header[4];
    model->config.num_heads = NH = model_header[5];
    model->config.channels = C = model_header[6];
    model->config.padded_vocab_size = Vp = model_header[7];
    printf("[GPT-2]\n");
    printf("max_seq_len: %zu\n", maxT);
    printf("vocab_size: %zu\n", V);
    printf("padded_vocab_size: %zu\n", Vp);
    printf("num_layers: %zu\n", L);
    printf("num_heads: %zu\n", NH);
    printf("channels: %zu\n", C);

    // allocate space for all the parameters and read them in
    fill_in_parameter_sizes(model->param_sizes,  model->config);

    // count the number of parameters
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += model->param_sizes[i];
    }
    printf("num_parameters: %zu\n", num_parameters);
    model->num_parameters = num_parameters;

    // read in all the parameters from file
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes);
    freadCheck(model->params_memory, sizeof(float), num_parameters, model_file);
    fcloseCheck(model_file);

    // other inits
    model->acts_memory = NULL;
    model->grads_memory = NULL;
    model->m_memory = NULL;
    model->v_memory = NULL;
    model->grads_acts_memory = NULL;
    model->inputs = NULL;
    model->targets = NULL;
    model->batch_size = 0;
    model->seq_len = 0;
    model->mean_loss = -1.0f; // -1.0f will designate no loss
}

#ifndef TESTING
// if we are TESTING (see test_gpt2.c), we'll skip the int main below

// main training loop
int main() {

  // build the GPT-2 model from a checkpoint
  GPT2 model;
  gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

  const char* tiny_stories_train = "dev/data/tinystories/TinyStories_train.bin";
  const char* tiny_stories_val = "dev/data/tinystories/TinyStories_val.bin";
  const char* tiny_shakespeare_train = "dev/data/tinyshakespeare/tiny_shakespeare_train.bin";
  const char* tiny_shakespeare_val = "dev/data/tinyshakespeare/tiny_shakespeare_val.bin";\
  // POSIX 系统调用（来自 <unistd.h>），用来检查文件是否存在 / 是否有某种权限。
  const char* train_tokens = access(tiny_shakespeare_train, F_OK) != -1 ? tiny_shakespeare_train : tiny_stories_train;
  const char* val_tokens = access(tiny_shakespeare_val, F_OK) != -1 ? tiny_shakespeare_val : tiny_stories_val;

  int B = 4; // batch size 4 (i.e. 4 independent token sequences will be trained on)
  int T = 64; // sequence length 64 (i.e. each sequence is 64 tokens long). must be <= maxT, which is 1024 for GPT-2
  DataLoader train_loader, val_loader;
  dataloader_init(&train_loader, train_tokens, B, T, 0, 1, 1);
  dataloader_init(&val_loader, val_tokens, B, T, 0, 1, 0);
  printf("train dataset num_batches: %zu\n", train_loader.num_tokens / (B*T));
  printf("val dataset num_batches: %zu\n", val_loader.num_tokens / (B*T));
  int val_num_batches = 5;
}
#endif