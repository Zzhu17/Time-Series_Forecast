import torch
import torch.nn as nn
from typing import Any, Optional, Dict

# 从独立的 layers.py 文件中导入所有需要的“零件”
from models.informer.layers import (
    DataEmbedding, 
    Encoder, 
    EncoderLayer, 
    Decoder, 
    DecoderLayer, 
    InformerOutputHead
)

# 辅助函数，用于从配置中安全地获取值
def get_conf_val(config: Any, key: str, default: Any = None) -> Any:
    """更健壮的配置获取函数，支持字典和对象两种形式。"""
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)

def _get_informer_config_from_main(main_config: Any) -> Dict[str, Any]:
    """
    从主配置文件中安全地提取出 Informer 的特定配置字典。
    这使得配置结构更灵活，例如可以将Informer配置放在 'model_config.Informer' 下。
    """
    # 尝试从嵌套结构中获取
    informer_config = get_conf_val(main_config, 'model_config', {})
    if isinstance(informer_config, dict):
        informer_config = informer_config.get('Informer', {})
    
    # 如果顶层直接就是模型配置，也支持
    if not informer_config:
        return main_config if isinstance(main_config, dict) else vars(main_config)
        
    return informer_config


class Informer(nn.Module):
    """
    Informer 模型主体。
    """
    def __init__(self, input_dim: int, out_features: int, d_model: int, n_heads: int, 
                 e_layers: int, d_layers: int, d_ff: int, dropout: float, 
                 attn_type: str = 'prob', factor: int = 5, device: str = 'cpu', 
                 dtype: torch.dtype = torch.float32, **kwargs):
        super().__init__()
        self.device = device
        self.dtype = dtype

        # 编码器部分
        self.enc_embedding = DataEmbedding(input_dim, d_model, dropout).to(device, dtype)
        self.encoder = Encoder(
            [
                EncoderLayer(d_model, n_heads, d_ff, dropout, attn_type, factor)
                for _ in range(e_layers)
            ],
            norm_layer=None,
            d_model=d_model
        ).to(device, dtype)

        # 解码器部分
        self.dec_embedding = DataEmbedding(input_dim, d_model, dropout).to(device, dtype)
        self.decoder = Decoder(
            [
                DecoderLayer(d_model, n_heads, d_ff, dropout, attn_type, factor)
                for _ in range(d_layers)
            ],
            norm_layer=None,
            d_model=d_model
        ).to(device, dtype)

        # 输出头
        self.output_head = InformerOutputHead(d_model, out_features).to(device, dtype)

    def forward(self, x_enc: torch.Tensor, x_dec: torch.Tensor, 
                enc_self_mask: Optional[torch.Tensor] = None, 
                dec_self_mask: Optional[torch.Tensor] = None, 
                dec_enc_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_enc = x_enc.to(self.device, self.dtype)
        x_dec = x_dec.to(self.device, self.dtype)
        
        enc_out = self.enc_embedding(x_enc)
        enc_out = self.encoder(enc_out, mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec)
        dec_out = self.decoder(dec_out, enc_out, tgt_mask=dec_self_mask, memory_mask=dec_enc_mask)
        
        final_out = self.output_head(dec_out)
        
        return final_out


def build_informer_model(config: Any) -> Informer:
    """
    【增强版】工厂函数：根据 config 创建 Informer 模型。
    - 增强了配置提取的健壮性。
    - 增加了关键参数的验证。
    - 优化了维度推断逻辑。
    """
    # 1. 首先，从可能嵌套的 config 中提取出 Informer 专属的配置字典
    informer_cfg = _get_informer_config_from_main(config)

    # 2. 推断输入和输出维度 (更清晰的逻辑)
    input_dim = get_conf_val(informer_cfg, 'enc_in')
    if input_dim is None:
        feature_cols = get_conf_val(informer_cfg, 'feature_cols', ['value'])
        if not isinstance(feature_cols, list) or not feature_cols:
             raise ValueError("'feature_cols' 必须是一个非空列表。")
        input_dim = len(feature_cols)

    out_features = get_conf_val(informer_cfg, 'c_out')
    if out_features is None:
        # 在主配置中寻找 target_col
        target_col = get_conf_val(config, 'target_col', get_conf_val(config, 'value_col', 'value'))
        out_features = 1 if isinstance(target_col, str) else len(target_col)

    # 3. 收集所有模型参数，并设置合理的默认值
    model_params = {
        'input_dim': input_dim,
        'out_features': out_features,
        'd_model': get_conf_val(informer_cfg, 'd_model', 512),
        'n_heads': get_conf_val(informer_cfg, 'n_heads', 8),
        'e_layers': get_conf_val(informer_cfg, 'e_layers', 2),
        'd_layers': get_conf_val(informer_cfg, 'd_layers', 1),
        'd_ff': get_conf_val(informer_cfg, 'd_ff', 2048),
        'dropout': get_conf_val(informer_cfg, 'dropout', 0.05),
        'attn_type': get_conf_val(informer_cfg, 'attn', 'prob'),
        'factor': get_conf_val(informer_cfg, 'factor', 5),
        'device': get_conf_val(config, 'device', 'cpu'), # device 通常是全局配置
    }

    # 4. 【新增】关键参数验证
    if model_params['d_model'] % model_params['n_heads'] != 0:
        raise ValueError(f"d_model ({model_params['d_model']}) 必须能被 n_heads ({model_params['n_heads']}) 整除。")

    # 5. 使用解包语法清晰地创建模型实例
    model = Informer(**model_params)
    
    return model
