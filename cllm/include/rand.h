//
// Created by twqb on 9/30/26.
//

#ifndef LLM_C_RANDH_H
#define LLM_C_RANDH_H

#include <math.h>

#define MERSENNE_STATE_M 397u
#define MERSENNE_STATE_N 624u

#define LMASK 0x7ffffffful
#define UMASK 0x80000000ul

// Copyright(c) Makoto Matsumoto and Takuji Nishimura

// This implementation follows PyTorch so that we are numerically identical when running verification tests.

typedef struct {
  unsigned long long seed_;
  int left_;
  unsigned int next_;
  unsigned int state_[MERSENNE_STATE_N];
  unsigned int MATRIX_A[2];
} mt19937_state;

void manual_seed(mt19937_state* state, unsigned int seed) {
  state->MATRIX_A[0] = 0x0u;
  state->MATRIX_A[1] = 0x9908b0df;
  state->state_[0] = seed & 0xffffffff;
  for (unsigned int j = 1; j < MERSENNE_STATE_N; j++) {
    state->state_[j] = 1812433253 * (state->state_[j - 1] ^ (state->state_[j - 1] >> 30)) + j;
    state->state_[j] &= 0xffffffff;
  }
  state->left_ = 1;
  state->next_ = 0;
}

void next_state(mt19937_state* state) {
  state->left_ = MERSENNE_STATE_N;
  state->next_ = 0;
  unsigned int y, j;
  for (j = 0; j < MERSENNE_STATE_N - MERSENNE_STATE_M; j++) {
    y = (state->state_[j] & UMASK) | (state->state_[j + 1] & LMASK);
    state->state_[j] = state->state_[j + MERSENNE_STATE_M] ^ (y >> 1) ^ state->MATRIX_A[y & 0x1];
  }
  for (; j < MERSENNE_STATE_N - 1; j++) {
    y = (state->state_[j] & UMASK) | (state->state_[j + 1] & LMASK);
    state->state_[j] = state->state_[j + (MERSENNE_STATE_M - MERSENNE_STATE_N)] ^ (y >> 1) ^ state->MATRIX_A[y & 0x1];
  }
  y = (state->state_[MERSENNE_STATE_N - 1] & UMASK) | (state->state_[0] & LMASK);
  state->state_[MERSENNE_STATE_N - 1] = state->state_[MERSENNE_STATE_M - 1] ^ (y >> 1) ^ state->MATRIX_A[y & 0x1];
}

unsigned int randint32(mt19937_state* state) {
  if (!state) return 0;
  if (state->MATRIX_A[0] != 0 || state->MATRIX_A[1] != 0x9908b0df) manual_seed(state, 5489); // auto-initialize
  if (--state->left_ <= 0) {
    next_state(state);
  }
  unsigned int y = state->state_[state->next_++];
  y ^= y >> 11;
  y ^= (y << 7) & 0x9d2c5680;
  y ^= (y << 15) & 0xefc60000;
  y ^= y >> 18;
  return y;
}

void init_identity_permutation(int *data, int numel) {
  for (int i = 0; i < numel; i++) {
    data[i] = i;
  }
}

void random_permutation(int* data, int numel, mt19937_state* state) {
  for (int i = numel - 1; i > 0; i--) {
    // pick an index j in [0, i] with equal probability
    int j = randint32(state) % (i + 1);
    // swap i <-> j
    int tmp = data[i];
    data[i] = data[j];
    data[j] = tmp;
  }
}

#endif //LLM_C_RANDH_H