import torch
import torch.nn as nn

# Gracefully handle missing triton dependency for Mac environments
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:
    @triton.jit
    def _mask_aware_fused_scan_kernel(
        events_ptr,          # [B, max_events, 3]
        event_mask_ptr,      # [B, max_events]
        A_log_ptr,           # [Dim, D_State]
        B_proj_ptr,          # [Dim, D_State]
        h_out_ptr,           # [B, Dim, D_State]
        B,
        max_events,
        Dim,
        D_State,
        events_b_stride,
        events_e_stride,
        mask_b_stride,
        mask_e_stride,
        h_out_b_stride,
        h_out_d_stride,
        BLOCK_DIM: tl.constexpr,
        BLOCK_DSTATE: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        
        # Setup SRAM Block Masks
        offsets_dim = tl.arange(0, BLOCK_DIM)
        offsets_dstate = tl.arange(0, BLOCK_DSTATE)
        
        dim_mask = offsets_dim < Dim
        dstate_mask = offsets_dstate < D_State
        h_mask = dim_mask[:, None] & dstate_mask[None, :]
        
        # Pointers and strides
        h_offsets = offsets_dim[:, None] * h_out_d_stride + offsets_dstate[None, :]
        
        # 1. Initialize Biological Hidden State completely in SRAM
        h = tl.zeros((BLOCK_DIM, BLOCK_DSTATE), dtype=tl.float32)
        last_t = tl.zeros((BLOCK_DIM,), dtype=tl.float32)
        
        # Load global decay A and projection B into SRAM
        A_log = tl.load(A_log_ptr + h_offsets, mask=h_mask, other=0.0)
        A = -tl.exp(A_log)
        B_proj = tl.load(B_proj_ptr + h_offsets, mask=h_mask, other=0.0)
        
        event_base = events_ptr + batch_idx * events_b_stride
        mask_base = event_mask_ptr + batch_idx * mask_b_stride
        
        # 2. One Blistering Fast Sweep over the chronological events
        for i in range(max_events):
            mask_ptr = mask_base + i * mask_e_stride
            is_valid = tl.load(mask_ptr)
            
            if is_valid:
                val_ptr = event_base + i * events_e_stride + 0
                sensor_id_ptr = event_base + i * events_e_stride + 1
                timestamp_ptr = event_base + i * events_e_stride + 2
                
                val = tl.load(val_ptr)
                sensor_id = tl.load(sensor_id_ptr).to(tl.int32)
                timestamp = tl.load(timestamp_ptr)
                
                sensor_mask = (offsets_dim == sensor_id)[:, None] & h_mask
                
                # Dynamic Delta T for the specific sensor
                # Extract the scalar last_t for the active sensor
                last_t_sensor = tl.sum(tl.where(offsets_dim == sensor_id, last_t, 0.0), axis=0)
                dt = timestamp - last_t_sensor
                
                # Continuous Time Discretization Update
                A_bar = tl.exp(A * dt)
                B_bar = ((A_bar - 1.0) / (A - 1e-8)) * B_proj * val
                
                h_new = A_bar * h + B_bar
                
                # Apply orthogonal routing (only update the active sensor's manifold)
                h = tl.where(sensor_mask, h_new, h)
                last_t = tl.where(offsets_dim == sensor_id, timestamp, last_t)

        # 3. Write final biological state out to slow VRAM
        h_out_base = h_out_ptr + batch_idx * h_out_b_stride
        tl.store(h_out_base + h_offsets, h, mask=h_mask)


class MaskAwareFusedScan(torch.autograd.Function):
    """
    Torch Autograd wrapper for the Triton Fused Scan Kernel.
    """
    @staticmethod
    def forward(ctx, events, event_mask, A_log, B_proj):
        # inputs must be contiguous for Triton to safely stride
        events = events.contiguous()
        event_mask = event_mask.contiguous()
        A_log = A_log.contiguous()
        B_proj = B_proj.contiguous()
        
        B_batch, max_events, _ = events.shape
        Dim, D_State = A_log.shape
        
        if not HAS_TRITON:
            raise RuntimeError("Triton is not installed on this system. Cannot run the Fused Scan Kernel.")
        
        # Calculate optimal block sizes for SRAM
        BLOCK_DIM = triton.next_power_of_2(Dim)
        BLOCK_DSTATE = triton.next_power_of_2(D_State)
        
        h_out = torch.zeros(B_batch, Dim, D_State, device=events.device, dtype=torch.float32)
        
        grid = (B_batch,)
        
        _mask_aware_fused_scan_kernel[grid](
            events,
            event_mask,
            A_log,
            B_proj,
            h_out,
            B_batch,
            max_events,
            Dim,
            D_State,
            events.stride(0), events.stride(1),
            event_mask.stride(0), event_mask.stride(1),
            h_out.stride(0), h_out.stride(1),
            BLOCK_DIM=BLOCK_DIM,
            BLOCK_DSTATE=BLOCK_DSTATE,
        )
        
        ctx.save_for_backward(events, event_mask, A_log, B_proj, h_out)
        return h_out

    @staticmethod
    def backward(ctx, grad_h_out):
        """
        Gradient derivation requires either retaining the full sequence of h_t
        in VRAM (expensive) or recomputing the forward pass sequentially 
        within the backward kernel (efficient but complex).
        """
        raise NotImplementedError("Backward pass for Triton fused scan is not yet implemented.")

def apply_fused_scan(events, event_mask, A_log, B_proj):
    return MaskAwareFusedScan.apply(events, event_mask, A_log, B_proj)
