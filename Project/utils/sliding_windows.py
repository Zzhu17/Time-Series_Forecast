import numpy as np
from typing import Tuple, Dict, Any

def create_windows_for_ml(
    data_values: np.ndarray, 
    input_steps: int, 
    output_steps: int, 
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    【通用函数】为传统机器学习和标准深度学习模型 (如 LSTM, GRU) 创建滑动窗口。
    
    这个函数生成标准的监督学习样本 (X, y)，其中 X 是历史序列，y 是未来序列。
    它原生支持单变量和多变量数据。

    Args:
        data_values (np.ndarray): 已经归一化好的、形状为 (n_samples, n_features) 的数据。
        input_steps (int): 用作输入的历史时间步数 (例如 LSTM 的 seq_len)。
        output_steps (int): 需要预测的未来时间步数 (例如 pred_len)。
        stride (int): 窗口滑动的步长。默认为1，表示逐个时间步滑动。

    Returns:
        Tuple[np.ndarray, np.ndarray]:
        - X: 输入特征，形状为 (n_windows, input_steps, n_features)。
        - y: 目标标签，形状为 (n_windows, output_steps, n_features)。
    """
    n_samples = data_values.shape[0]
    window_size = input_steps + output_steps

    if n_samples < window_size:
        raise ValueError(
            f"数据长度不足。需要至少 {window_size} 个样本才能创建窗口 "
            f"(input_steps={input_steps} + output_steps={output_steps}), 但只有 {n_samples} 个样本。"
        )

    X_list, y_list = [], []
    
    for i in range(0, n_samples - window_size + 1, stride):
        # 输入窗口: [i, i + input_steps)
        input_window = data_values[i : i + input_steps]
        
        # 输出窗口: [i + input_steps, i + input_steps + output_steps)
        output_window = data_values[i + input_steps : i + input_steps + output_steps]
        
        X_list.append(input_window)
        y_list.append(output_window)

    X = np.array(X_list)
    y = np.array(y_list)

    return X, y


def create_windows_for_informer(
    data_values: np.ndarray, 
    config: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    【专用函数】为 Informer 模型创建其独特的输入窗口。

    此函数根据 Informer 的 seq_len, label_len, pred_len 参数，
    生成编码器输入(x_enc)、解码器输入(x_dec)、目标标签(y_label)和残差模型特征(x_feature)。

    Args:
        data_values (np.ndarray): 已经归一化好的、形状为 (n_samples, n_features) 的数据。
        config (Dict[str, Any]): Informer 的模型配置字典，必须包含 'seq_len', 'label_len', 'pred_len'。

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        - x_enc: 编码器输入, shape (n_windows, seq_len, n_features)
        - x_dec: 解码器输入, shape (n_windows, label_len + pred_len, n_features)
        - y_label: 真实标签, shape (n_windows, label_len + pred_len, n_features)
        - x_feature: 用于残差模型的特征, shape (n_windows, seq_len * n_features)
    """
    seq_len = config['seq_len']
    label_len = config['label_len']
    pred_len = config['pred_len']
    
    n_samples = data_values.shape[0]
    window_size = seq_len + pred_len
    
    if n_samples < window_size:
        raise ValueError(
            f"数据长度不足。需要至少 {window_size} 个样本才能创建窗口 "
            f"(seq_len={seq_len} + pred_len={pred_len}), 但只有 {n_samples} 个样本。"
        )

    x_enc_list, x_dec_list, y_label_list = [], [], []

    for i in range(n_samples - window_size + 1):
        # 编码器输入: [i, i + seq_len)
        enc_start = i
        enc_end = i + seq_len
        x_enc_list.append(data_values[enc_start:enc_end, :])

        # 解码器输入: [i + seq_len - label_len, i + seq_len + pred_len)
        # 这是 Informer 的一个关键点：解码器输入包含一部分历史(label_len)和未来(pred_len)
        dec_start = enc_end - label_len
        dec_end = enc_end + pred_len
        x_dec_list.append(data_values[dec_start:dec_end, :])
        
        # 目标标签 (y_label) 在Informer中与解码器输入(x_dec)是相同的，
        # 因为模型的目标是预测出完整的 x_dec 序列。
        y_label_list.append(data_values[dec_start:dec_end, :])

    x_enc = np.array(x_enc_list)
    x_dec = np.array(x_dec_list)
    y_label = np.array(y_label_list)

    # 创建用于残差模型的特征
    # 将每个编码器窗口的 (seq_len, n_features) 展平为 (seq_len * n_features)
    x_feature = x_enc.reshape(x_enc.shape[0], -1)

    return x_enc, x_dec, y_label, x_feature
