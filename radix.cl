/*
Copyright 2011-2012 Andreas Kloeckner
Copyright 2008-2011 NVIDIA Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Derived from thrust/detail/backend/cuda/detail/fast_scan.inl
within the Thrust project, https://code.google.com/p/thrust/

Direct link to thrust source:
https://code.google.com/p/thrust/source/browse/thrust/detail/backend/cuda/detail/fast_scan.inl
*/
#define REQD_WG_SIZE(X,Y,psc_Z) __attribute__((reqd_work_group_size(X, Y, psc_Z)))
#define psc_WG_SIZE 256

#define psc_SCAN_EXPR(a, b, across_seg_boundary) scan_t_add(a, b, across_seg_boundary)

typedef struct {
  unsigned c00;
  unsigned c01;
  unsigned c10;
  unsigned c11;
} pyopencl_sort_scan_uint32_2bits_t;

//CL//
#define psc_INPUT_EXPR(i) (scan_t_from_value(dkey[i], base_bit, i))

typedef pyopencl_sort_scan_uint32_2bits_t scan_t;
typedef unsigned key_t;
typedef unsigned index_t;

// #define DEBUG
#ifdef DEBUG
#define dbg_printf(ARGS) printf ARGS
#else
#define dbg_printf(ARGS) /* */
#endif

index_t get_count(scan_t s, int mnr) {
  return ((mnr < 2) ? ((mnr < 1) ? s.c00 : s.c01) : ((mnr < 3) ? s.c10 : s.c11));
}

#define BIN_NR(key_arg) ((key_arg >> base_bit) & 3)

scan_t scan_t_neutral() {
  scan_t result;
  result.c00 = 0;
  result.c01 = 0;
  result.c10 = 0;
  result.c11 = 0;
  return result;
}

// considers bits (base_bit+bits-1, ..., base_bit)
scan_t scan_t_from_value(
    key_t key,
    int base_bit,
    int i)
{
  // extract relevant bit range
  key_t bin_nr = BIN_NR(key);

  dbg_printf(("i: %d key:%d bin_nr:%d\n", i, key, bin_nr));

  scan_t result;
  result.c00 = (bin_nr == 0);
  result.c01 = (bin_nr == 1);
  result.c10 = (bin_nr == 2);
  result.c11 = (bin_nr == 3);
  return result;
}

scan_t scan_t_add(scan_t a, scan_t b, bool across_seg_boundary) {
  b.c00 = a.c00 + b.c00;
  b.c01 = a.c01 + b.c01;
  b.c10 = a.c10 + b.c10;
  b.c11 = a.c11 + b.c11;
  return b;
}


typedef pyopencl_sort_scan_uint32_2bits_t psc_scan_type;
typedef int psc_index_type;

// NO_SEG_BOUNDARY is the largest representable integer in psc_index_type.
// This assumption is used in code below.
#define NO_SEG_BOUNDARY 2147483647

#define psc_K 8

// #define psc_DEBUG
#ifdef psc_DEBUG
#define pycl_printf(psc_ARGS) printf psc_ARGS
#else
#define pycl_printf(psc_ARGS) /* */
#endif

__kernel
REQD_WG_SIZE(psc_WG_SIZE, 1, 1)
void scan_scan_intervals_1(
    __global unsigned *dkey, __global unsigned *dval, __global unsigned *sorted_dkey, __global unsigned *sorted_dval, int base_bit,
    __global psc_scan_type *restrict psc_partial_scan_buffer,
    const psc_index_type N,
    const psc_index_type psc_interval_size
    , __global psc_scan_type *restrict psc_interval_results
    )
{
  // index psc_K in first dimension used for psc_carry storage
  // Avoid bank conflicts by adding a single 32-bit psc_value to the size of
  // the scan type.
  struct __attribute__ ((__packed__)) psc_wrapped_scan_type
  {
    psc_scan_type psc_value;
    int psc_dummy;
  };
  __local struct psc_wrapped_scan_type psc_ldata[psc_K + 1][psc_WG_SIZE + 1];

  // {{{ declare local data for input_fetch_exprs if any of them are stenciled
  // }}}

  const psc_index_type psc_interval_begin = psc_interval_size * get_group_id(0);
  const psc_index_type psc_interval_end   = min(psc_interval_begin + psc_interval_size, N);

  const psc_index_type psc_unit_size  = psc_K * psc_WG_SIZE;

  psc_index_type psc_unit_base = psc_interval_begin;

  for(; psc_unit_base + psc_unit_size <= psc_interval_end; psc_unit_base += psc_unit_size) {
    // {{{ psc_carry out input_fetch_exprs
    // (if there are ones that need to be fetched into local)

    pycl_printf(("after input_fetch_exprs\n"));

    // }}}

    // {{{ read a unit's worth of data from psc_global

    for(psc_index_type psc_k = 0; psc_k < psc_K; psc_k++) {
      const psc_index_type psc_offset = psc_k*psc_WG_SIZE + get_local_id(0);
      const psc_index_type psc_read_i = psc_unit_base + psc_offset;

      psc_scan_type psc_scan_value = psc_INPUT_EXPR(psc_read_i);

      const psc_index_type psc_o_mod_k = psc_offset % psc_K;
      const psc_index_type psc_o_div_k = psc_offset / psc_K;
      psc_ldata[psc_o_mod_k][psc_o_div_k].psc_value = psc_scan_value;
    }

    pycl_printf(("after read from psc_global\n"));

    // }}}

    // {{{ psc_carry in from previous unit, if applicable
    if (get_local_id(0) == 0 && psc_unit_base != psc_interval_begin) {
      psc_ldata[0][0].psc_value = psc_SCAN_EXPR(psc_ldata[psc_K][psc_WG_SIZE - 1].psc_value, psc_ldata[0][0].psc_value, false);
    }

    pycl_printf(("after psc_carry-in\n"));

    // }}}
    barrier(CLK_LOCAL_MEM_FENCE);

    // {{{ scan along psc_k (sequentially in each work item)
    psc_scan_type psc_sum = psc_ldata[0][get_local_id(0)].psc_value;

    for(psc_index_type psc_k = 1; psc_k < psc_K; psc_k++) {
      psc_scan_type psc_tmp = psc_ldata[psc_k][get_local_id(0)].psc_value;
      psc_index_type psc_seq_i = psc_unit_base + psc_K*get_local_id(0) + psc_k;

      psc_sum = psc_SCAN_EXPR(psc_sum, psc_tmp, false);
      psc_ldata[psc_k][get_local_id(0)].psc_value = psc_sum;
    }

    pycl_printf(("after scan along psc_k\n"));

    // }}}

    // store psc_carry in out-of-bounds (padding) array entry (index psc_K) in
    // the psc_K direction
    psc_ldata[psc_K][get_local_id(0)].psc_value = psc_sum;

    barrier(CLK_LOCAL_MEM_FENCE);

    // {{{ tree-based local parallel scan

    // This tree-based scan works as follows:
    // - Each work item adds the previous item to its current state
    // - barrier
    // - Each work item adds in the item from two positions to the left
    // - barrier
    // - Each work item adds in the item from four positions to the left
    // ...
    // At the end, each item has summed all prior items.

    // across psc_k groups, along local id
    // (uses out-of-bounds psc_k=psc_K array entry for storage)

    psc_scan_type psc_val = psc_ldata[psc_K][get_local_id(0)].psc_value;

    for (uint s = 1; s <= 256 ; s <<= 1) {
    // {{{ reads from local allowed, writes to local not allowed
      if (get_local_id(0) >= s) {
	psc_scan_type psc_tmp = psc_ldata[psc_K][get_local_id(0) - s].psc_value;
	psc_val = psc_SCAN_EXPR(psc_tmp, psc_val, false);
      }
      // }}}
      barrier(CLK_LOCAL_MEM_FENCE);

      // {{{ writes to local allowed, reads from local not allowed
      psc_ldata[psc_K][get_local_id(0)].psc_value = psc_val;

      // }}}
      barrier(CLK_LOCAL_MEM_FENCE);
    }

    pycl_printf(("after tree scan\n"));

    // }}}

    // {{{ update local values

    if (get_local_id(0) > 0) {
      psc_sum = psc_ldata[psc_K][get_local_id(0) - 1].psc_value;

      for (psc_index_type psc_k = 0; psc_k < psc_K; psc_k++) {
	psc_scan_type psc_tmp = psc_ldata[psc_k][get_local_id(0)].psc_value;
	psc_ldata[psc_k][get_local_id(0)].psc_value = psc_SCAN_EXPR(psc_sum, psc_tmp, false);
      }
    }

    pycl_printf(("after local update\n"));

    // }}}
    barrier(CLK_LOCAL_MEM_FENCE);

    // {{{ write data
    {
      // work hard with index math to achieve contiguous 32-bit stores
      __global int *psc_dest = (__global int *) (psc_partial_scan_buffer + psc_unit_base);

      const psc_index_type psc_scan_types_per_int = 4;

      for (uint s = 0; s < 8192; s += 256) {
	psc_index_type psc_linear_index = s + get_local_id(0);
	psc_index_type psc_linear_scan_data_idx = psc_linear_index / psc_scan_types_per_int;
	psc_index_type remainder = psc_linear_index - psc_linear_scan_data_idx * psc_scan_types_per_int;

	__local int *psc_src = (__local int *) &(psc_ldata[psc_linear_scan_data_idx % psc_K][psc_linear_scan_data_idx / psc_K].psc_value); 
	psc_dest[psc_linear_index] = psc_src[remainder];
      }
    }

    pycl_printf(("after write\n"));

    // }}}
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (psc_unit_base < psc_interval_end) {
    // {{{ psc_carry out input_fetch_exprs
    // (if there are ones that need to be fetched into local)
    pycl_printf(("after input_fetch_exprs\n"));

    // }}}

    // {{{ read a unit's worth of data from psc_global

    for(psc_index_type psc_k = 0; psc_k < psc_K; psc_k++) {
      const psc_index_type psc_offset = psc_k*psc_WG_SIZE + get_local_id(0);
      const psc_index_type psc_read_i = psc_unit_base + psc_offset;

      if (psc_read_i < psc_interval_end) { 
	psc_scan_type psc_scan_value = psc_INPUT_EXPR(psc_read_i);

	const psc_index_type psc_o_mod_k = psc_offset % psc_K;
	const psc_index_type psc_o_div_k = psc_offset / psc_K;
	psc_ldata[psc_o_mod_k][psc_o_div_k].psc_value = psc_scan_value;

      }
    }

    pycl_printf(("after read from psc_global\n"));

    // }}}

    // {{{ psc_carry in from previous unit, if applicable
    if (get_local_id(0) == 0 && psc_unit_base != psc_interval_begin) {
      psc_ldata[0][0].psc_value = psc_SCAN_EXPR(psc_ldata[psc_K][psc_WG_SIZE - 1].psc_value, psc_ldata[0][0].psc_value, false);
    }

    pycl_printf(("after psc_carry-in\n"));

    // }}}
    barrier(CLK_LOCAL_MEM_FENCE);

    // {{{ scan along psc_k (sequentially in each work item)
    psc_scan_type psc_sum = psc_ldata[0][get_local_id(0)].psc_value;

    const psc_index_type psc_offset_end = psc_interval_end - psc_unit_base;

    for(psc_index_type psc_k = 1; psc_k < psc_K; psc_k++) {
      if (psc_K * get_local_id(0) + psc_k < psc_offset_end) {
	psc_scan_type psc_tmp = psc_ldata[psc_k][get_local_id(0)].psc_value;
	psc_index_type psc_seq_i = psc_unit_base + psc_K*get_local_id(0) + psc_k;

	psc_sum = psc_SCAN_EXPR(psc_sum, psc_tmp, false); 
	psc_ldata[psc_k][get_local_id(0)].psc_value = psc_sum;
      }
    }

    pycl_printf(("after scan along psc_k\n"));

    // }}}

    // store psc_carry in out-of-bounds (padding) array entry (index psc_K) in
    // the psc_K direction
    psc_ldata[psc_K][get_local_id(0)].psc_value = psc_sum;

    barrier(CLK_LOCAL_MEM_FENCE);

    // {{{ tree-based local parallel scan

    // This tree-based scan works as follows:
    // - Each work item adds the previous item to its current state
    // - barrier
    // - Each work item adds in the item from two positions to the left
    // - barrier
    // - Each work item adds in the item from four positions to the left
    // ...
    // At the end, each item has summed all prior items.

    // across psc_k groups, along local id
    // (uses out-of-bounds psc_k=psc_K array entry for storage)

    psc_scan_type psc_val = psc_ldata[psc_K][get_local_id(0)].psc_value;

    for (uint s = 1; s <= 256 ; s <<= 1) {
      // {{{ reads from local allowed, writes to local not allowed
      if (get_local_id(0) >= s) {
	psc_scan_type psc_tmp = psc_ldata[psc_K][get_local_id(0) - s].psc_value;
	if (psc_K*get_local_id(0) < psc_offset_end) {
	  psc_val = psc_SCAN_EXPR(psc_tmp, psc_val, false);
	}
      }
      // }}}
      barrier(CLK_LOCAL_MEM_FENCE);

      // {{{ writes to local allowed, reads from local not allowed
      psc_ldata[psc_K][get_local_id(0)].psc_value = psc_val;
      // }}}
      barrier(CLK_LOCAL_MEM_FENCE);
    }

    pycl_printf(("after tree scan\n"));

    // }}}

    // {{{ update local values

    if (get_local_id(0) > 0) {
      psc_sum = psc_ldata[psc_K][get_local_id(0) - 1].psc_value;

      for(psc_index_type psc_k = 0; psc_k < psc_K; psc_k++) {
	if (psc_K * get_local_id(0) + psc_k < psc_offset_end) {
	  psc_scan_type psc_tmp = psc_ldata[psc_k][get_local_id(0)].psc_value;
	  psc_ldata[psc_k][get_local_id(0)].psc_value = psc_SCAN_EXPR(psc_sum, psc_tmp, false);
	}
      }
    }

    pycl_printf(("after local update\n"));

    // }}}
    barrier(CLK_LOCAL_MEM_FENCE);

    // {{{ write data
    {
      // work hard with index math to achieve contiguous 32-bit stores
      __global int *psc_dest = (__global int *) (psc_partial_scan_buffer + psc_unit_base);

      const psc_index_type psc_scan_types_per_int = 4;

      for (uint s = 0; s < 8192; s += 256) {
	if (s + get_local_id(0) < psc_scan_types_per_int*(psc_interval_end - psc_unit_base)) {
	  psc_index_type psc_linear_index = s + get_local_id(0);
	  psc_index_type psc_linear_scan_data_idx = psc_linear_index / psc_scan_types_per_int;
	  psc_index_type remainder = psc_linear_index - psc_linear_scan_data_idx * psc_scan_types_per_int;

	  __local int *psc_src = (__local int *) &(psc_ldata[psc_linear_scan_data_idx % psc_K][psc_linear_scan_data_idx / psc_K].psc_value);
	  psc_dest[psc_linear_index] = psc_src[remainder];
	}
      }
    }

    pycl_printf(("after write\n"));

    // }}}
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // write interval psc_sum
  if (get_local_id(0) == 0) {
    psc_interval_results[get_group_id(0)] = psc_partial_scan_buffer[psc_interval_end - 1];
  }
}

//CL//
#define psc_INPUT_EXPR(i) (interval_sums[i])

__kernel
REQD_WG_SIZE(psc_WG_SIZE, 1, 1)
void scan_scan_intervals_0(
    __global unsigned *dkey, __global unsigned *dval, __global unsigned *sorted_dkey, __global unsigned *sorted_dval, int base_bit, __global pyopencl_sort_scan_uint32_2bits_t *interval_sums,
    __global psc_scan_type *restrict psc_partial_scan_buffer,
    const psc_index_type N,
    const psc_index_type psc_interval_size
    )
{
  // index psc_K in first dimension used for psc_carry storage
  // Avoid bank conflicts by adding a single 32-bit psc_value to the size of
  // the scan type.
  struct __attribute__ ((__packed__)) psc_wrapped_scan_type
  {
    psc_scan_type psc_value;
    int psc_dummy;
  };
  __local struct psc_wrapped_scan_type psc_ldata[psc_K + 1][psc_WG_SIZE + 1];

  // {{{ declare local data for input_fetch_exprs if any of them are stenciled
  // }}}

  const psc_index_type psc_interval_begin = psc_interval_size * get_group_id(0);
  const psc_index_type psc_interval_end   = min(psc_interval_begin + psc_interval_size, N);

  const psc_index_type psc_unit_size  = psc_K * psc_WG_SIZE;
  psc_index_type psc_unit_base = psc_interval_begin;

  for(; psc_unit_base + psc_unit_size <= psc_interval_end; psc_unit_base += psc_unit_size) {
    // {{{ psc_carry out input_fetch_exprs
    // (if there are ones that need to be fetched into local)

    pycl_printf(("after input_fetch_exprs\n"));

    // }}}

    // {{{ read a unit's worth of data from psc_global
    for(psc_index_type psc_k = 0; psc_k < psc_K; psc_k++) {
      const psc_index_type psc_offset = psc_k*psc_WG_SIZE + get_local_id(0);
      const psc_index_type psc_read_i = psc_unit_base + psc_offset;

      psc_scan_type psc_scan_value = psc_INPUT_EXPR(psc_read_i);

      const psc_index_type psc_o_mod_k = psc_offset % psc_K;
      const psc_index_type psc_o_div_k = psc_offset / psc_K;
      psc_ldata[psc_o_mod_k][psc_o_div_k].psc_value = psc_scan_value;
    }

    pycl_printf(("after read from psc_global\n"));

    // }}}

    // {{{ psc_carry in from previous unit, if applicable
    if (get_local_id(0) == 0 && psc_unit_base != psc_interval_begin) {
      psc_ldata[0][0].psc_value = psc_SCAN_EXPR(psc_ldata[psc_K][psc_WG_SIZE - 1].psc_value, psc_ldata[0][0].psc_value, false);
    }

    pycl_printf(("after psc_carry-in\n"));

    // }}}
    barrier(CLK_LOCAL_MEM_FENCE);

    // {{{ scan along psc_k (sequentially in each work item)
    psc_scan_type psc_sum = psc_ldata[0][get_local_id(0)].psc_value;

    for(psc_index_type psc_k = 1; psc_k < psc_K; psc_k++) {
	psc_scan_type psc_tmp = psc_ldata[psc_k][get_local_id(0)].psc_value;
	psc_index_type psc_seq_i = psc_unit_base + psc_K*get_local_id(0) + psc_k;

	psc_sum = psc_SCAN_EXPR(psc_sum, psc_tmp, false);
	psc_ldata[psc_k][get_local_id(0)].psc_value = psc_sum;
    }

    pycl_printf(("after scan along psc_k\n"));

    // }}}

    // store psc_carry in out-of-bounds (padding) array entry (index psc_K) in
    // the psc_K direction
    psc_ldata[psc_K][get_local_id(0)].psc_value = psc_sum;

    barrier(CLK_LOCAL_MEM_FENCE);

    // {{{ tree-based local parallel scan

    // This tree-based scan works as follows:
    // - Each work item adds the previous item to its current state
    // - barrier
    // - Each work item adds in the item from two positions to the left
    // - barrier
    // - Each work item adds in the item from four positions to the left
    // ...
    // At the end, each item has summed all prior items.

    // across psc_k groups, along local id
    // (uses out-of-bounds psc_k=psc_K array entry for storage)

    psc_scan_type psc_val = psc_ldata[psc_K][get_local_id(0)].psc_value;

    for (uint s = 1; s <= 256 ; s <<= 1) {
      // {{{ reads from local allowed, writes to local not allowed

      if (get_local_id(0) >= s) {
	psc_scan_type psc_tmp = psc_ldata[psc_K][get_local_id(0) - s].psc_value;
	psc_val = psc_SCAN_EXPR(psc_tmp, psc_val, false);
      }

      // }}}
      barrier(CLK_LOCAL_MEM_FENCE);

      // {{{ writes to local allowed, reads from local not allowed
      psc_ldata[psc_K][get_local_id(0)].psc_value = psc_val;

      // }}}
      barrier(CLK_LOCAL_MEM_FENCE);
    }

    pycl_printf(("after tree scan\n"));

    // }}}

    // {{{ update local values

    if (get_local_id(0) > 0) {
      psc_sum = psc_ldata[psc_K][get_local_id(0) - 1].psc_value;

      for (psc_index_type psc_k = 0; psc_k < psc_K; psc_k++) {
	psc_scan_type psc_tmp = psc_ldata[psc_k][get_local_id(0)].psc_value;
	psc_ldata[psc_k][get_local_id(0)].psc_value = psc_SCAN_EXPR(psc_sum, psc_tmp, false);
      }
    }

    pycl_printf(("after local update\n"));

    // }}}
    barrier(CLK_LOCAL_MEM_FENCE);

    // {{{ write data

    {
      // work hard with index math to achieve contiguous 32-bit stores
      __global int *psc_dest = (__global int *) (psc_partial_scan_buffer + psc_unit_base);

      const psc_index_type psc_scan_types_per_int = 4;

      for (uint s = 0; s < 8192; s += 256) {
	psc_index_type psc_linear_index = s + get_local_id(0);
	psc_index_type psc_linear_scan_data_idx = psc_linear_index / psc_scan_types_per_int;
	psc_index_type remainder = psc_linear_index - psc_linear_scan_data_idx * psc_scan_types_per_int;

	__local int *psc_src = (__local int *) &(psc_ldata[psc_linear_scan_data_idx % psc_K][psc_linear_scan_data_idx / psc_K].psc_value);
	psc_dest[psc_linear_index] = psc_src[remainder];
      }
    }

    pycl_printf(("after write\n"));

    // }}}
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (psc_unit_base < psc_interval_end) {
    // {{{ psc_carry out input_fetch_exprs
    // (if there are ones that need to be fetched into local)

    pycl_printf(("after input_fetch_exprs\n"));

    // }}}

    // {{{ read a unit's worth of data from psc_global
    for(psc_index_type psc_k = 0; psc_k < psc_K; psc_k++) {
      const psc_index_type psc_offset = psc_k*psc_WG_SIZE + get_local_id(0);
      const psc_index_type psc_read_i = psc_unit_base + psc_offset;

      if (psc_read_i < psc_interval_end) {
	psc_scan_type psc_scan_value = psc_INPUT_EXPR(psc_read_i);

	const psc_index_type psc_o_mod_k = psc_offset % psc_K;
	const psc_index_type psc_o_div_k = psc_offset / psc_K;
	psc_ldata[psc_o_mod_k][psc_o_div_k].psc_value = psc_scan_value;

      }
    }

    pycl_printf(("after read from psc_global\n"));

    // }}}

    // {{{ psc_carry in from previous unit, if applicable


    if (get_local_id(0) == 0 && psc_unit_base != psc_interval_begin) {
      psc_ldata[0][0].psc_value = psc_SCAN_EXPR(psc_ldata[psc_K][psc_WG_SIZE - 1].psc_value, psc_ldata[0][0].psc_value, false);
    }

    pycl_printf(("after psc_carry-in\n"));

    // }}}
    barrier(CLK_LOCAL_MEM_FENCE);

    // {{{ scan along psc_k (sequentially in each work item)
    psc_scan_type psc_sum = psc_ldata[0][get_local_id(0)].psc_value;
    const psc_index_type psc_offset_end = psc_interval_end - psc_unit_base;

    for(psc_index_type psc_k = 1; psc_k < psc_K; psc_k++) {
      if (psc_K * get_local_id(0) + psc_k < psc_offset_end) {
	psc_scan_type psc_tmp = psc_ldata[psc_k][get_local_id(0)].psc_value;
	psc_index_type psc_seq_i = psc_unit_base + psc_K*get_local_id(0) + psc_k;

	psc_sum = psc_SCAN_EXPR(psc_sum, psc_tmp, false);
	psc_ldata[psc_k][get_local_id(0)].psc_value = psc_sum;
      }
    }

    pycl_printf(("after scan along psc_k\n"));

    // }}}

    // store psc_carry in out-of-bounds (padding) array entry (index psc_K) in
    // the psc_K direction
    psc_ldata[psc_K][get_local_id(0)].psc_value = psc_sum;

    barrier(CLK_LOCAL_MEM_FENCE);

    // {{{ tree-based local parallel scan

    // This tree-based scan works as follows:
    // - Each work item adds the previous item to its current state
    // - barrier
    // - Each work item adds in the item from two positions to the left
    // - barrier
    // - Each work item adds in the item from four positions to the left
    // ...
    // At the end, each item has summed all prior items.

    // across psc_k groups, along local id
    // (uses out-of-bounds psc_k=psc_K array entry for storage)

    psc_scan_type psc_val = psc_ldata[psc_K][get_local_id(0)].psc_value;

    for (uint s = 1; s <= 256 ; s <<= 1) {
      // {{{ reads from local allowed, writes to local not allowed

      if (get_local_id(0) >= s) {
	psc_scan_type psc_tmp = psc_ldata[psc_K][get_local_id(0) - s].psc_value;
	if (psc_K*get_local_id(0) < psc_offset_end) {
	  psc_val = psc_SCAN_EXPR(psc_tmp, psc_val, false);
	}
      }

      // }}}
      barrier(CLK_LOCAL_MEM_FENCE);

      // {{{ writes to local allowed, reads from local not allowed
      psc_ldata[psc_K][get_local_id(0)].psc_value = psc_val;

      // }}}
      barrier(CLK_LOCAL_MEM_FENCE);
    }

    pycl_printf(("after tree scan\n"));

    // }}}

    // {{{ update local values

    if (get_local_id(0) > 0) {
      psc_sum = psc_ldata[psc_K][get_local_id(0) - 1].psc_value;

      for(psc_index_type psc_k = 0; psc_k < psc_K; psc_k++) {
	if (psc_K * get_local_id(0) + psc_k < psc_offset_end) {
	  psc_scan_type psc_tmp = psc_ldata[psc_k][get_local_id(0)].psc_value;
	  psc_ldata[psc_k][get_local_id(0)].psc_value = psc_SCAN_EXPR(psc_sum, psc_tmp, false);
	}
      }
    }

    pycl_printf(("after local update\n"));

    // }}}
    barrier(CLK_LOCAL_MEM_FENCE);

    // {{{ write data
    {
      // work hard with index math to achieve contiguous 32-bit stores
      __global int *psc_dest = (__global int *) (psc_partial_scan_buffer + psc_unit_base);

      const psc_index_type psc_scan_types_per_int = 4;

      for (uint s = 0; s < 8192; s += 256) {
	if (s + get_local_id(0) < psc_scan_types_per_int*(psc_interval_end - psc_unit_base)) {
	  psc_index_type psc_linear_index = s + get_local_id(0);
	  psc_index_type psc_linear_scan_data_idx = psc_linear_index / psc_scan_types_per_int;
	  psc_index_type remainder = psc_linear_index - psc_linear_scan_data_idx * psc_scan_types_per_int;

	  __local int *psc_src = (__local int *) &(psc_ldata[psc_linear_scan_data_idx % psc_K][psc_linear_scan_data_idx / psc_K].psc_value);
	  psc_dest[psc_linear_index] = psc_src[remainder];
	}
      }
    }

    pycl_printf(("after write\n"));

    // }}}
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // write interval psc_sum
}

//CL//
#define psc_INPUT_EXPR(i) (scan_t_from_value(dkey[i], base_bit, i))

__kernel
REQD_WG_SIZE(psc_WG_SIZE, 1, 1)
void scan_final_update(
    __global unsigned *dkey, __global unsigned *dval, __global unsigned *sorted_dkey, __global unsigned *sorted_dval, int base_bit,
    const psc_index_type N,
    const psc_index_type psc_interval_size,
    __global psc_scan_type *restrict psc_interval_results,
    __global psc_scan_type *restrict psc_partial_scan_buffer
    )
{
  const psc_index_type psc_interval_begin = psc_interval_size * get_group_id(0);
  const psc_index_type psc_interval_end = min(psc_interval_begin + psc_interval_size, N);

  // psc_carry from last interval
  psc_scan_type psc_carry = scan_t_neutral();
  if (get_group_id(0) != 0)
    psc_carry = psc_interval_results[get_group_id(0) - 1];

  psc_scan_type last_item = psc_interval_results[get_num_groups(0)-1];

  // {{{ no look-behind ('prev_item' not in output_statement -> simpler)
  psc_index_type psc_update_i = psc_interval_begin+get_local_id(0);

  for(; psc_update_i < psc_interval_end; psc_update_i += psc_WG_SIZE) {
    psc_scan_type psc_partial_val = psc_partial_scan_buffer[psc_update_i];
    psc_scan_type item = psc_SCAN_EXPR(psc_carry, psc_partial_val, false);
    psc_index_type i = psc_update_i;

    { //CL//
      {
	key_t key = dkey[i];
	key_t my_bin_nr = BIN_NR(key);

	index_t previous_bins_size = 0;
	previous_bins_size += (my_bin_nr > 0) ? last_item.c00 : 0;
	previous_bins_size += (my_bin_nr > 1) ? last_item.c01 : 0;
	previous_bins_size += (my_bin_nr > 2) ? last_item.c10 : 0;
	previous_bins_size += (my_bin_nr > 3) ? last_item.c11 : 0;

	index_t tgt_idx = previous_bins_size + get_count(item, my_bin_nr) - 1;

	sorted_dkey[tgt_idx] = dkey[i];
	sorted_dval[tgt_idx] = dval[i];
      }
      ; }
  }

  // }}}
}


#define pycl_offsetof(st, m) \
  ((size_t) ((__local char *) &(dummy.m) - (__local char *)&dummy ))


__kernel void get_size_and_offsets(__global size_t *result)
{
  result[0] = sizeof(pyopencl_sort_scan_uint32_2bits_t);
  __local pyopencl_sort_scan_uint32_2bits_t dummy;
  result[1] = pycl_offsetof(pyopencl_sort_scan_uint32_2bits_t, c00);
  result[2] = pycl_offsetof(pyopencl_sort_scan_uint32_2bits_t, c01);
  result[3] = pycl_offsetof(pyopencl_sort_scan_uint32_2bits_t, c10);
  result[4] = pycl_offsetof(pyopencl_sort_scan_uint32_2bits_t, c11);
}

