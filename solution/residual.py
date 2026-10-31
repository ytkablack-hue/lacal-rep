"""Own Triton kernels: the attention residual add fused into the next RMSNorm.

In a Qwen2 decoder layer the stock code does
    h = residual + attention_output          # one elementwise kernel (fp32 here)
    y = post_attention_layernorm(h)          # RMSNorm
and the backward pass adds the gradient of the residual path to the gradient of the norm input in
one more elementwise kernel. Here a single kernel reads residual and attention output once, writes
both `h` (it is the next residual) and `y`, and the backward kernel adds the two gradients while it
computes the norm's input gradient.

Same numbers as the separate ops: the add is the same fp32 addition (rounded to the storage dtype
before the norm reads it, exactly as when the sum is a tensor in memory), the norm is the maths of
`rmsnorm.py`, and the backward sum is the same two-term fp32 addition autograd performed.

Safety. Qwen2DecoderLayer.forward is replaced only if its source, parsed with `ast`, has exactly the
expected nine statements (residual, input norm, attention, add, residual, post norm, MLP, add,
return). The attention call is rebuilt from that very call, so every argument the stock layer passes
is passed. Any other layout (another transformers version) leaves the stock forward untouched.
"""

import ast
import inspect
import textwrap

import torch
import triton
import triton.language as tl

from .rmsnorm import _launch_config


@triton.jit
def add_rmsnorm_fwd_kernel(r_ptr, b_ptr, w_ptr, s_ptr, y_ptr, rstd_ptr, n_cols, eps,
                           BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    offs = row.to(tl.int64) * n_cols + cols

    r = tl.load(r_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    s_store = (r + b).to(s_ptr.dtype.element_ty)  # the sum as the stock add would store it
    tl.store(s_ptr + offs, s_store, mask=mask)

    s = s_store.to(tl.float32)
    var = tl.sum(s * s, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    xn = (s * rstd).to(s_ptr.dtype.element_ty).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + offs, (xn * w).to(y_ptr.dtype.element_ty), mask=mask)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def add_rmsnorm_bwd_kernel(dy_ptr, ds_ptr, s_ptr, w_ptr, rstd_ptr, dr_ptr, n_cols,
                           HAS_DS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    offs = row.to(tl.int64) * n_cols + cols

    dy = tl.load(dy_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(s_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row)

    xn = s * rstd
    g = dy * w
    c = tl.sum(g * xn, axis=0) / n_cols
    # the norm's input gradient, rounded to the storage dtype as the separate RMSNorm backward does
    dx = (rstd * (g - xn * c)).to(dr_ptr.dtype.element_ty).to(tl.float32)
    if HAS_DS:
        dx = dx + tl.load(ds_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(dr_ptr + offs, dx.to(dr_ptr.dtype.element_ty), mask=mask)


class AddRMSNormFunction(torch.autograd.Function):
    """(residual, branch) -> (residual + branch, rmsnorm(residual + branch) * weight)."""

    @staticmethod
    def forward(ctx, residual, branch, weight, eps):
        ctx.set_materialize_grads(False)
        shape = residual.shape
        r2 = residual.reshape(-1, shape[-1]).contiguous()
        b2 = branch.reshape(-1, shape[-1]).contiguous()
        weight = weight.contiguous()
        n_rows, n_cols = r2.shape
        # the sum has the dtype the stock `residual + branch` gives (fp32 residual + fp16 branch -> fp32)
        s_dtype = torch.promote_types(r2.dtype, b2.dtype)
        s = torch.empty(r2.shape, dtype=s_dtype, device=r2.device)
        y = torch.empty((n_rows, n_cols), dtype=torch.promote_types(s_dtype, weight.dtype),
                        device=r2.device)
        rstd = torch.empty(n_rows, dtype=torch.float32, device=r2.device)
        block, num_warps = _launch_config(n_cols)
        add_rmsnorm_fwd_kernel[(n_rows,)](r2, b2, weight, s, y, rstd, n_cols, eps, BLOCK=block,
                                          num_warps=num_warps)
        s_out = s.view(shape)
        ctx.save_for_backward(s_out, weight, rstd)
        ctx.shape = shape
        ctx.residual_dtype, ctx.branch_dtype = residual.dtype, branch.dtype
        return s_out, y.view(shape)

    @staticmethod
    def backward(ctx, ds, dy):
        s, weight, rstd = ctx.saved_tensors
        n_cols = s.shape[-1]
        if dy is None:  # the normalised output was not used: only the sum's gradient flows
            return (ds.to(ctx.residual_dtype) if ds is not None else None,
                    ds.to(ctx.branch_dtype) if ds is not None else None, None, None)
        s2 = s.reshape(-1, n_cols)
        dy2 = dy.reshape(-1, n_cols).contiguous()
        has_ds = ds is not None
        ds2 = ds.reshape(-1, n_cols).contiguous() if has_ds else dy2
        dr = torch.empty_like(s2)
        block, num_warps = _launch_config(n_cols)
        add_rmsnorm_bwd_kernel[(s2.shape[0],)](dy2, ds2, s2, weight, rstd, dr, n_cols,
                                               HAS_DS=has_ds, BLOCK=block, num_warps=num_warps)
        dw = None
        if ctx.needs_input_grad[2]:  # frozen in this task, kept for correctness
            xn = s2.float() * rstd[:, None]
            dw = (dy2.float() * xn).sum(0).to(weight.dtype)
        grad = dr.view(ctx.shape)
        # the same gradient flows to both addends, in each one's own dtype (as autograd's add does)
        return grad.to(ctx.residual_dtype), grad.to(ctx.branch_dtype), dw, None


def add_rmsnorm(residual, branch, weight, eps):
    return AddRMSNormFunction.apply(residual, branch, weight, eps)


def _layer_plan(forward):
    """Check that `forward` is the expected decoder-layer code. Returns a function that builds the
    keyword arguments of the stock `self.self_attn(...)` call from a call of the layer, or None."""
    try:
        node = ast.parse(textwrap.dedent(inspect.getsource(forward))).body[0]
        signature = inspect.signature(forward)
    except Exception:
        return None
    if not isinstance(node, ast.FunctionDef):
        return None
    body = [s for s in node.body if not (isinstance(s, ast.Expr)
                                         and isinstance(s.value, ast.Constant))]
    if len(body) != 9:
        return None

    def is_name(n, ident):
        return isinstance(n, ast.Name) and n.id == ident

    def single_target(stmt):
        return stmt.targets[0] if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 else None

    def self_call(value, attr, one_arg=True):
        ok = (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
              and value.func.attr == attr and is_name(value.func.value, "self"))
        if ok and one_arg:
            ok = (len(value.args) == 1 and is_name(value.args[0], "hidden_states")
                  and not value.keywords)
        return ok

    def adds_residual(value):
        return (isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add)
                and is_name(value.left, "residual") and is_name(value.right, "hidden_states"))

    t = [single_target(s) for s in body]
    attn_target = t[2]
    attn_ok = is_name(attn_target, "hidden_states") or (
        isinstance(attn_target, ast.Tuple) and attn_target.elts
        and is_name(attn_target.elts[0], "hidden_states"))
    structure = (
        is_name(t[0], "residual") and is_name(body[0].value, "hidden_states")
        and is_name(t[1], "hidden_states") and self_call(body[1].value, "input_layernorm")
        and attn_ok and self_call(body[2].value, "self_attn", one_arg=False)
        and is_name(t[3], "hidden_states") and adds_residual(body[3].value)
        and is_name(t[4], "residual") and is_name(body[4].value, "hidden_states")
        and is_name(t[5], "hidden_states") and self_call(body[5].value, "post_attention_layernorm")
        and is_name(t[6], "hidden_states") and self_call(body[6].value, "mlp")
        and is_name(t[7], "hidden_states") and adds_residual(body[7].value)
        and isinstance(body[8], ast.Return) and is_name(body[8].value, "hidden_states"))
    if not structure:
        return None

    call = body[2].value
    if call.args:
        return None
    pairs, var_keyword = [], None
    for keyword in call.keywords:
        if not isinstance(keyword.value, ast.Name):
            return None
        if keyword.arg is None:
            var_keyword = keyword.value.id
        else:
            pairs.append((keyword.arg, keyword.value.id))
    if ("hidden_states", "hidden_states") not in pairs:
        return None
    parameters = signature.parameters
    for _, var in pairs:
        if var != "hidden_states" and var not in parameters:
            return None
    if var_keyword is not None and (var_keyword not in parameters or parameters[
            var_keyword].kind is not inspect.Parameter.VAR_KEYWORD):
        return None

    def attention_kwargs(layer, hidden_states, args, kwargs):
        bound = signature.bind(layer, hidden_states, *args, **kwargs)  # TypeError: not our call
        bound.apply_defaults()
        out = {key: bound.arguments[var] for key, var in pairs if key != "hidden_states"}
        if var_keyword is not None:
            out.update(bound.arguments.get(var_keyword) or {})
        return out

    return attention_kwargs


def patch_qwen2_decoder_residual():
    """Qwen2DecoderLayer.forward with the residual add fused into the next RMSNorm, when the stock
    layer has the expected layout. Safe if transformers is a stub."""
    try:
        from transformers.models.qwen2 import modeling_qwen2
        cls = modeling_qwen2.Qwen2DecoderLayer
        original = cls.forward
    except Exception:
        return
    if getattr(cls, "_own_residual", False):
        return
    attention_kwargs = _layer_plan(original)
    if attention_kwargs is None:
        return

    def forward(self, hidden_states, *args, **kwargs):
        norm = getattr(self, "post_attention_layernorm", None)
        if (not torch.is_tensor(hidden_states) or not hidden_states.is_cuda
                or not torch.is_tensor(getattr(norm, "weight", None))):
            return original(self, hidden_states, *args, **kwargs)
        try:
            call_kwargs = attention_kwargs(self, hidden_states, args, kwargs)
        except TypeError:
            return original(self, hidden_states, *args, **kwargs)

        residual = hidden_states
        attn = self.self_attn(hidden_states=self.input_layernorm(hidden_states), **call_kwargs)
        attn = attn[0] if isinstance(attn, tuple) else attn
        if (attn.shape == residual.shape and attn.is_cuda and attn.is_floating_point()
                and residual.is_floating_point()):
            eps = getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6))
            total, normed = add_rmsnorm(residual, attn, norm.weight, eps)
        else:
            total = residual + attn
            normed = norm(total)
        return total + self.mlp(normed)

    cls._own_residual_original = original
    cls.forward = forward
    cls._own_residual = True
