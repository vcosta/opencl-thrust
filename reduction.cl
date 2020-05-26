/*
Copyright 2010 Andreas Kloeckner

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

Based on code/ideas by Mark Harris <mharris@nvidia.com>.
None of the original source code remains.
*/
#define GROUP_SIZE 1024
typedef OUT_TYPE out_type;
typedef IN_TYPE in_type;

__kernel
void NAME_PREFIX(_reduce_kernel)(
  __global out_type *out, __global in_type *x,
  uint seq_count, uint n)
{
  __local out_type ldata[GROUP_SIZE];

  uint lid = get_local_id(0);

  uint i = get_group_id(0)*GROUP_SIZE*seq_count + lid;

  out_type acc = 0;
  for (uint s = 0; s < seq_count; ++s) {
    if (i >= n)
      break;
    acc = REDUCE(acc, READ_AND_MAP(i));

    i += GROUP_SIZE;
  }

  ldata[lid] = acc;

  barrier(CLK_LOCAL_MEM_FENCE);

  for(uint s = get_local_size(0) >> 1; s >= 32; s >>= 1) {
    if (lid < s) {
      ldata[lid] = REDUCE( ldata[lid], ldata[lid + s] );
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid < 32) {
    __local volatile out_type *lvdata = ldata;

    lvdata[lid] = REDUCE( lvdata[lid], lvdata[lid + 16]);
    lvdata[lid] = REDUCE( lvdata[lid], lvdata[lid + 8]);
    lvdata[lid] = REDUCE( lvdata[lid], lvdata[lid + 4]);
    lvdata[lid] = REDUCE( lvdata[lid], lvdata[lid + 2]);
    lvdata[lid] = REDUCE( lvdata[lid], lvdata[lid + 1]);
  }
;
  if (lid == 0) out[get_group_id(0)] = ldata[0];
}

