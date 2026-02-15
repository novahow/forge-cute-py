import math
from typing import Type, Optional
from functools import partial

import torch

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Int64, Float32, const_expr
import operator

@cute.jit
def make_reduction_buffer_layout(tv_layout: cute.Layout):
    warps_per_row = tv_layout.shape[0][0] // cute.arch.WARP_SIZE
    thread_rows = tv_layout.shape[0][1]
    return (thread_rows, warps_per_row)


@cute.jit
def row_reduce(
    x: cute.TensorSSA,
    threads_per_row: cutlass.Constexpr[int],
    reduction_buffer: Optional[cute.Tensor] = None,
) -> [Float32]:
    sum_x = cute.arch.warp_reduction(
        x.reduce(cute.ReductionOp.ADD, init_val=0, reduction_profile=0),
        operator.add,
        threads_in_group=min(threads_per_row, cute.arch.WARP_SIZE),
    )
    rows_per_block, warps_per_row = reduction_buffer.shape
    lane_idx, warp_idx = cute.arch.lane_idx(), cute.arch.warp_idx()
    row_idx, col_idx = warp_idx // warps_per_row, warp_idx % warps_per_row
    if lane_idx == 0:
        reduction_buffer[row_idx, col_idx] = sum_x
    cute.arch.barrier()
    sum_x = 0.0
    if lane_idx < warps_per_row:
        sum_x = reduction_buffer[row_idx, lane_idx]

    sum_x_final = cute.arch.warp_reduction(sum_x, operator.add)
    return sum_x_final

@cute.jit
def online_softmax_reduce(
    x: cute.TensorSSA,
    threads_per_row: cutlass.Constexpr[int],
    reduction_buffer_mx: Optional[cute.Tensor] = None,
    reduction_buffer_sum: Optional[cute.Tensor] = None,
) -> [Float32, Float32, cute.TensorSSA]:
    # TODO: use f32x2_to_i64 instead of 2 reduction buffers
    assert x.dtype == Float32, "x must be of type Float32"
    # get per-warp max
    max_x = cute.arch.warp_reduction(
        x.reduce(cute.ReductionOp.MAX, init_val=-Float32.inf, reduction_profile=0),
        cute.arch.fmax,
        threads_in_group=min(threads_per_row, cute.arch.WARP_SIZE),
    )
    log2_e = math.log2(math.e)
    exp_x = cute.math.exp2(x * log2_e - (max_x * log2_e), fastmath=True)
    # get per-warp expsum
    sum_exp_x = cute.arch.warp_reduction(
        exp_x.reduce(cute.ReductionOp.ADD, init_val=0.0, reduction_profile=0),
        operator.add,
        threads_in_group=min(threads_per_row, cute.arch.WARP_SIZE),
    )
    rows_per_block, warps_per_row = reduction_buffer_mx.shape
    lane_idx, warp_idx = cute.arch.lane_idx(), cute.arch.warp_idx()
    row_idx, col_idx = warp_idx // warps_per_row, warp_idx % warps_per_row
    if lane_idx == 0:
        reduction_buffer_mx[row_idx, col_idx] = max_x
        reduction_buffer_sum[row_idx, col_idx] = sum_exp_x
    cute.arch.barrier()
    max_x_single_warp = -Float32.inf
    sum_exp_x = 0.0
    # load per-warp values into registers
    if lane_idx < warps_per_row:
        max_x_single_warp, sum_exp_x = reduction_buffer_mx[row_idx, lane_idx], reduction_buffer_sum[row_idx, lane_idx]

    max_x_final = cute.arch.warp_reduction(max_x_single_warp, cute.arch.fmax)
    sum_exp_x *= cute.math.exp(max_x_single_warp - max_x_final, fastmath=True)
    sum_exp_x = cute.arch.warp_reduction(sum_exp_x, operator.add)
    exp_x *= cute.math.exp(max_x - max_x_final, fastmath=True)
    max_x = max_x_final

    return max_x, sum_exp_x, exp_x

class Softmax:
    def __init__(self, dtype: Type[cutlass.Numeric], N: int, online_softmax: bool = True):
        self.N = N
        self.dtype = dtype

    def _threads_per_row(self):
        N = self.N
        for limit, threads in [(64, 8), (128, 16), (3072, 32), (6144, 64), (16384, 128)]:
            if N <= limit:
                return threads
        return 256

    def _set_cluster_n(self):
        raise NotImplementedError

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mO: cute.Tensor,
        # stream: cuda.CUstream,
        num_threads: cutlass.Constexpr[int],
        threads_per_row: cutlass.Constexpr[int],
        # variant: int=2,
    ):
        assert mX.element_type == self.dtype
        largest_dtype_width = const_expr(max(t.element_type.width for t in [mX, mO]))
        assert num_threads % threads_per_row == 0, "non-divisible"
        largest_dtype_width = const_expr(max(t.element_type.width for t in [mX, mO]))
        vecsize = 128 // largest_dtype_width
        cpy_atom = cute.make_copy_atom(
            # cpasync.CopyG2SOp(),
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=largest_dtype_width * vecsize,
        )
        rows_in_block = num_threads // threads_per_row

        layout_t = cute.make_ordered_layout(
            (rows_in_block, threads_per_row),
            order=(1, 0),
        )
        REGISTER_LIMIT = 64
        maxN = min(num_threads * REGISTER_LIMIT, self.N)
        layout_v = cute.make_ordered_layout((1, vecsize), order=(1,0))
        tiled_copy = cute.make_tiled_copy_tv(cpy_atom, layout_t, layout_v)
        tiler_mn = (rows_in_block, maxN)
        blocks = cute.ceil_div(mX.shape[0], tiler_mn[0])
        print(num_threads, tiler_mn)
        
        num_threads = tiled_copy.size
        print(num_threads)
        self.kernel2(mX, mO, tiler_mn, tiled_copy, threads_per_row).launch(
            grid=[blocks, 1, 1],
            block=[num_threads, 1, 1],
            # stream=stream,
        )



    @cute.kernel
    def kernel2(
        self,
        mX: cute.Tensor,
        mO: cute.Tensor,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
    ):
        """
        tiler_mn: the shape this tiled_copy handles: (n_threads // threads_per_row, N)
        tiled_copy: t layout: (threads_per_row, n_threads // threads_per_row), v layout: (vecsize)
        """
        tv_layout = tiled_copy.layout_tv_tiled
        # cute.printf(tv_layout)
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        # shape = mX.shape
        tiler_coord = (bidx, None)
       
        gX = cute.local_tile(mX,  cute.select(tiler_mn, mode=[0,1]), cute.select(tiler_coord, mode=[0, 1])) # (rows_in_block, maxN, k)
        gO = cute.local_tile(mO, tiler_mn, (bidx, None))
        print("gX:", gX.layout, tiler_mn, tv_layout.shape)
        # cute.printf(gX)
        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            mX.element_type, 
            cute.make_ordered_layout(tiler_mn, order=(1,0)), 
            byte_alignment=mX.element_type.width
        )
        sO = smem.allocate_tensor(
            mO.element_type, 
            cute.make_ordered_layout(tiler_mn, order=(1,0)), 
            byte_alignment=mO.element_type.width
        )

        thr_copy = tiled_copy.get_slice(tidx)
        cpy_atom = cute.make_copy_atom(
            # cpasync.CopyG2SOp(),
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=mX.element_type.width * tv_layout.shape[-1],
        )
        reduction_buffer_shape = make_reduction_buffer_layout(tv_layout)
        reduction_buffer_mx = smem.allocate_tensor(
            cute.Float32,
            cute.make_ordered_layout(reduction_buffer_shape, order=(1,0)), 
            byte_alignment=cute.Float32.width
        )
        reduction_buffer_sum = smem.allocate_tensor(
            cute.Float32,
            cute.make_ordered_layout(reduction_buffer_shape, order=(1,0)), 
            byte_alignment=cute.Float32.width
        )

        max_x_final = -Float32.inf
        sum_exp_x_final = 0.0
        for k in cutlass.range(gX.shape[-1]):
            tXgX = thr_copy.partition_S(gX[None, None, k]) # ((vecsize, 1), 1, num_blocks_N)
            print("tXgX:", tXgX.layout, tiled_copy)
            tXsX = thr_copy.partition_D(sX)
            print("tXsX:", tXsX.layout, tiled_copy)
            tOgO = thr_copy.partition_S(gO[None, None, k])
            tXrX = cute.make_fragment_like(tXsX)
            tOrO = cute.make_fragment_like(tOgO)

            cute.copy(cpy_atom, tXgX, tXrX)
            x = tXrX.load().to(cute.Float32)

            # cute.autovec

            max_x, sum_exp_x, exp_x = online_softmax_reduce(
                x, 
                threads_per_row, 
                reduction_buffer_mx, 
                reduction_buffer_sum
            )
            cur_max = cute.arch.fmax(max_x, max_x_final)
            scale_old = cute.math.exp(max_x_final - cur_max, fastmath=True)
            scale_curr = cute.math.exp(max_x - cur_max, fastmath=True)
            max_x_final = cur_max
            sum_exp_x_final = sum_exp_x_final * scale_old + sum_exp_x * scale_curr
            y = x
            # store tensorSSA back to register
            # tOrO.store(y.to(tOrO.element_type))
            # cute.copy(cpy_atom, tOrO, tOgO)

        for k in range(gX.shape[-1]):
            tXgX = thr_copy.partition_S(gX[None, None, k]) # ((vecsize, 1), 1, num_blocks_N)
            tXsX = thr_copy.partition_D(sX)
            tOgO = thr_copy.partition_S(gO[None, None, k])
            tXrX = cute.make_fragment_like(tXsX)
            tOrO = cute.make_fragment_like(tOgO)

            cute.copy(cpy_atom, tXgX, tXrX)
            x = tXrX.load().to(cute.Float32)
            log2_e = math.log2(math.e)
            exp_x = cute.math.exp2(x * log2_e - (max_x_final * log2_e), fastmath=True)
            y = exp_x * cute.arch.rcp_approx(sum_exp_x_final)
            # store tensorSSA back to register
            tOrO.store(y.to(tOrO.element_type))
            cute.copy(cpy_atom, tOrO, tOgO)


class SoftmaxBackward:
    def __init__(self, dtype: Type[cutlass.Numeric], N: int):
        # 1 stage for computing dot product
        self.dtype = dtype
        self.N = N
        self.reduction_type = Float32

    def _threads_per_row(self):
        N = self.N
        for limit, threads in [(64, 8), (128, 16), (3072, 32), (6144, 64), (8192, 128)]:
            if N <= limit:
                return threads
        return 256

    def _set_cluster_n(self):
        raise NotImplementedError

    def _num_threads(self):
        return 128 if self.N <= 8192 else 256

    @cute.jit
    def __call__(
        self,
        mdY: cute.Tensor,
        mY: cute.Tensor,
        mdX: cute.Tensor,
        num_threads: cutlass.Constexpr[int],
        threads_per_row: cutlass.Constexpr[int],
        # stream: cuda.CUstream,
    ):
        assert mdY.element_type == self.dtype
        largest_dtype_width = const_expr(max(t.element_type.width for t in [mdY, mY, mdX]))
        vecsize = 128 // largest_dtype_width
        cpy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mdY.element_type,
            num_bits_per_copy=largest_dtype_width * vecsize,
        )
        rows_in_block = num_threads // threads_per_row

        layout_t = cute.make_ordered_layout(
            (rows_in_block, threads_per_row),
            order=(1, 0),
        )
        layout_v = cute.make_ordered_layout((1, vecsize), order=(1,0))
        tiled_copy = cute.make_tiled_copy_tv(cpy_atom, layout_t, layout_v)
        tiler_mn = (rows_in_block, self.N)
        blocks = cute.ceil_div(mdY.shape[0], tiler_mn[0])
        
        num_threads = tiled_copy.size
        self.kernel(mdY, mY, mdX, tiler_mn, tiled_copy, threads_per_row).launch(
            grid=[blocks, 1, 1],
            block=[num_threads, 1, 1],
            # stream=stream,
        )
        

    @cute.kernel
    def kernel(
        self,
        mdY: cute.Tensor,
        mY: cute.Tensor,
        mdX: cute.Tensor,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
    ):
        """
        tiler_mn: the shape this tiled_copy handles: (n_threads // threads_per_row, N)
        tiled_copy: t layout: (threads_per_row, n_threads // threads_per_row), v layout: (vecsize)
        """
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        tv_layout = tiled_copy.layout_tv_tiled

        gdY = cute.local_tile(mdY, tiler_mn, (bidx, 0))
        gY = cute.local_tile(mY, tiler_mn, (bidx, 0))
        gdX = cute.local_tile(mdX, tiler_mn, (bidx, 0))

        smem = cutlass.utils.SmemAllocator()
        sdY = smem.allocate_tensor(
            mdY.element_type, 
            cute.make_ordered_layout(tiler_mn, order=(1,0)), 
            byte_alignment=mdY.element_type.width
        )
        sY = smem.allocate_tensor(
            mY.element_type, 
            cute.make_ordered_layout(tiler_mn, order=(1,0)), 
            byte_alignment=mY.element_type.width
        )

        thr_copy = tiled_copy.get_slice(tidx)
        tGgG = thr_copy.partition_S(gdY)
        tGsG = thr_copy.partition_D(sdY)
        tdXgdX = thr_copy.partition_D(gdX)
        tYgY = thr_copy.partition_S(gY)
        tYsY = thr_copy.partition_D(sY)
        tGrG = cute.make_fragment_like(tGsG)
        tYrY = cute.make_fragment_like(tYsY)
        tXrX = cute.make_fragment_like(tdXgdX)


        cpy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mdY.element_type,
            num_bits_per_copy=mdY.element_type.width * tGgG.shape[0][0],
        )
        cute.copy(cpy_atom, tGgG, tGrG)
        cute.copy(cpy_atom, tYgY, tYrY)
        g_val = tGrG.load().to(cute.Float32)
        y_val = tYrY.load().to(cute.Float32)
        x = g_val * y_val

        reduction_buffer_shape = make_reduction_buffer_layout(tv_layout)
        reduction_buffer = smem.allocate_tensor(
            x.element_type,
            cute.make_ordered_layout(reduction_buffer_shape, order=(1,0)), 
            byte_alignment=x.element_type.width
        )
        row_yg = row_reduce(x, threads_per_row, reduction_buffer)
        dxi = y_val * (g_val - row_yg)
        tXrX.store(dxi.to(tXrX.element_type))
        cute.copy(cpy_atom, tXrX, tdXgdX)


if __name__ == '__main__':
    M = 4096
    N = 16 * 1024
    dtype = torch.float16

    x = torch.randn(M, N, device='cuda', dtype=dtype)
    out = torch.zeros(M, N, device='cuda', dtype=dtype)


    # Instantiate Reduction class
    # variant argument is integer in kernel definition but string in ops?
    # In kernel: __init__(self, dtype, N, variant: int)
    # Let's pass 1 for now implementation doesn't seem to use it for dispatch in __call__ yet

    reduction_op = Softmax(cutlass.Float16, N, 1) # variant=1

    # Determine correct grid/block dimensions if not handled by __call__
    # The __call__ implementation has:
    # self.kernel2(...).launch(grid=[blocks, 1, 1], block=[NUM_THREADS, 1, 1])
    # blocks = mX.shape[0] -> M

    from cutlass.cute.runtime import from_dlpack
    print(torch.finfo(x.dtype).bits)
    mX_cute = from_dlpack(x, assumed_align=torch.finfo(x.dtype).bits)
    mO_cute = from_dlpack(out, assumed_align=torch.finfo(out.dtype).bits)

    print("Calling Reduction kernel...")
    try:
        # , cute.runtime.make_fake_stream()
        reduction_op(mX_cute, mO_cute, 128, 64)
    except Exception as e:
        print(f"Kernel launch failed: {e}")
        import traceback
        traceback.print_exc()
        # return

    expected = torch.softmax(x, dim=1)
    
    print(f"Output shape: {out.shape}")
    print(f"Expected shape: {expected.shape}")
    
    if not torch.allclose(out, expected, atol=1e-4, rtol=1e-4):
        print("Mismatch found!")
        print(f"Max diff: {(out - expected).abs().max().item()}")
        print(f"Out[:10]: {out[:10]}")
        print(f"Exp[:10]: {expected[:10]}")
    else:
        print("Success! Forward output matches torch reference.")
        print(f"Out[0][:10]: {out[0][:10]}")
        print(f"Exp[0][:10]: {expected[0][:10]}")

    # Backward test
    print("\nTesting Backward pass...")
    dy = torch.randn(M, N, device='cuda', dtype=dtype)
    dx = torch.zeros(M, N, device='cuda', dtype=dtype)
    
    # We use the forward output 'out' as 'y'
    y = out.clone()
    
    mdY_cute = from_dlpack(dy, assumed_align=torch.finfo(dy.dtype).bits)
    mY_cute = from_dlpack(y, assumed_align=torch.finfo(y.dtype).bits)
    mdX_cute = from_dlpack(dx, assumed_align=torch.finfo(dx.dtype).bits)
    
    reduction_bwd_op = SoftmaxBackward(cutlass.Float16, N)
    
    try:
        reduction_bwd_op(mdY_cute, mY_cute, mdX_cute, 128, 64)
    except Exception as e:
        print(f"Backward Kernel launch failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
        
    # Torch reference backward
    x_ref = x.clone().requires_grad_(True)
    y_ref = torch.softmax(x_ref, dim=1)
    y_ref.backward(dy)
    expected_dx = x_ref.grad
    
    print(f"DX shape: {dx.shape}")
    if not torch.allclose(dx, expected_dx, atol=1e-3, rtol=1e-3):
        print("Backward Mismatch found!")
        print(f"Max diff: {(dx - expected_dx).abs().max().item()}")
        print(f"DX[0][:10]: {dx[0][:10]}")
        print(f"ExpDX[0][:10]: {expected_dx[0][:10]}")
    else:
        print("Success! Backward output matches torch reference.")
        print(f"DX[0][:10]: {dx[0][:10]}")
        print(f"ExpDX[0][:10]: {expected_dx[0][:10]}")
