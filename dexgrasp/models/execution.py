"""Opt-in execution optimizations without changing checkpoint parameter names."""


def configure_attention_precision(model):
    if model.cfg.self_attention_backend == 'centered_fp32':
        import torch
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False


def configure_keypoint_execution(model, cfg=None):
    configure_attention_precision(model)
    if cfg is None:
        return
    count = cfg.get('checkpoint_transformer_blocks', 0)
    if type(count) is not int or not 0 <= count <= len(model.blocks):
        raise ValueError('checkpoint_transformer_blocks must be between zero and the model depth')
    model.checkpoint_transformer_blocks = count
    for i, block in enumerate(model.blocks):
        block.checkpoint_direction_bias = i >= count and cfg['checkpoint_direction_bias']
    for encoder in (model.object_encoder, model.scene_encoder):
        if encoder is not None:
            encoder.fps_backend = cfg['fps_backend']
            encoder.select_tokens_before_slice = cfg['select_tokens_before_slice']
            if cfg['sparse_convolution_mode'] != 'default':
                from MinkowskiEngineBackend._C import ConvolutionMode
                mode = (ConvolutionMode.COPY_GEMM if cfg['sparse_convolution_mode'].startswith('copy')
                        else ConvolutionMode.DIRECT_GEMM)
                for module in encoder.modules():
                    if hasattr(module, 'convolution_mode'):
                        if cfg['sparse_convolution_mode'] != 'copy_hidden' or min(module.in_channels, module.out_channels) >= 64:
                            module.convolution_mode = mode
    if cfg['compile_transformer']:
        # Leave dynamic sparse coordinates and input validation outside Dynamo.
        # Module.compile mutates only the call path, preserving state_dict keys.
        # No CUDA graphs: MinkowskiEngine manages separate dynamic allocations.
        for block in model.blocks:
            block.compile(fullgraph=True, dynamic=False)
