import torch
import os
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import noisereduce as nr

def denoise_audio(audio_file):
    # 加载音频文件
    audio, sr = torchaudio.load(audio_file)

    # 从音频文件中截取一段只有噪声的部分
    noise = audio[0:10000]

    # 使用noisereduce库去除噪声
    reduced_audio = nr.reduce_noise(y=audio, y_noise=noise, sr=sr)

    # 返回去除噪声后的音频信号和采样率
    return reduced_audio, sr

def data_loader(args):
    kwopt = {'num_workers': 0, 'pin_memory': True}

    # 设置文件夹路径和其他参数
    audio_folder = 'langid_train/langid/train' # 替换为实际的音频文件夹路径

    # 读取数据并处理
    audio_files = sorted(os.listdir(audio_folder), key=lambda x: int(x[:-4]) if x.endswith('.wav') else float('inf'))

    # 添加路径到文件名
    audio_files_with_path = [os.path.join(audio_folder, file) for file in audio_files]

    # 定义目标采样率
    target_sample_rate = 48000

    # 定义目标长度
    target_length = 256

    # 对音频数据进行重采样、MFCC 转换和归一化
    features = []
    for audio_file in audio_files_with_path:
        # 对音频文件进行去噪处理
        waveform, sample_rate = denoise_audio(audio_file)
        # 没有去噪
        # waveform, sample_rate = torchaudio.load(audio_file)

        # 把音频信号从数组转换为张量，并且转换为浮点类型
        waveform = torch.from_numpy(waveform).float()

        # 定义重采样转换器
        resample = T.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resample(waveform)

        # 定义STFT
        stft = T.Spectrogram(
            n_fft=128,
            win_length=128,
            hop_length=64,
            center=True,
            pad_mode="reflect",
            power=None,
        )

        # 定义 MFCC 转换器
        mfcc = T.MFCC(
            sample_rate=target_sample_rate,
            n_mfcc=40,
            melkwargs={
                'n_fft': 128,
                'n_mels': 128,
                'hop_length': 512,
                # 添加 center 参数为 False，避免在音频信号两端进行填充
                'center': False
            }
        )

        # 计算 MFCC 特征
        feature = mfcc(waveform)
        # 对 MFCC 特征进行归一化
        feature = F.sliding_window_cmn(feature, cmn_window=600, norm_vars=True, center=False)

        # 计算stft，为了方便，生成的变量名不做改变
        # feature = stft(waveform)
        # 把STFT特征转换为幅度谱
        # feature = torch.abs(feature)
        # 对STFT特征进行归一化，并转换为分贝谱
        # feature = F.amplitude_to_DB(feature, multiplier=10.0, amin=1e-10, db_multiplier=0.0)


        # 获取音频文件的长度
        length = feature.shape[2]
        # 如果音频文件的长度小于目标长度，就进行填充
        if length < target_length:
            # 计算需要在两端填充的元素个数
            pad_left = (target_length - length) // 2
            pad_right = target_length - length - pad_left
            # 创建一个填充转换器，指定在两端填充的元素个数和填充值
            pad = torch.nn.ConstantPad1d((pad_left, pad_right), 0.0)
            # 对音频文件进行填充
            feature = pad(feature)
        # 如果音频文件的长度大于目标长度，就进行截断
        elif length > target_length:
            # 对音频文件进行截断，使其长度不超过目标长度
            feature = feature[:, :, :target_length]
        features.append(feature)

    # 读取标签文件，并添加到列表中
    with open('langid_train/langid/lab.txt', 'r') as f:
        labels = f.read().splitlines()

    # Before stacking, pad the tensors with zeros
    # 对列表中的张量进行转置，使第二维度变为第一维度
    features = [i.transpose(0, 1) for i in features]
    # 使用 pad_sequence 函数填充张量，并指定 batch_first 参数为 True，不指定 total_length 参数
    features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0.0)
    # 对输出张量进行转置，使第二维度变回原来的位置
    features = features.transpose(1, 2)


    # 把labels中的字符串拆分为单个字符的列表，并映射为整数，然后转换为张量
    # 定义一个字符到整数的映射字典
    char_to_int = {'C': 0, 'M': 1, 'E': 2}
    # 把labels中的字符串拆分为单个字符的列表
    labels = list(labels[0])
    # 把每个字符映射为一个整数
    labels = [char_to_int[char] for char in labels]
    # 把整数列表转换为一个张量
    labels = torch.tensor(labels)
    # 定义一个类别的数量，即字符的种类数
    num_classes = len(char_to_int)
    # 创建一个形状为（batch_size, num_classes）的零张量，用于存储one-hot编码
    one_hot_labels = torch.zeros((len(labels), num_classes))
    # 根据labels中的整数，将one_hot_labels中对应位置的元素设为1
    one_hot_labels.scatter_(1, labels.unsqueeze(1), 1)

    # 将音频数据和标签数据组合成一个数据集对象
    dataset = TensorDataset(features, one_hot_labels)

    # 计算训练集和测试集的大小，按照 7:3 的比例划分
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    test_size = total_size - train_size

    # 随机划分训练集和测试集
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 创建训练数据加载器和测试数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 返回 train_dataloader 和 test_dataloader
    return train_dataloader, test_dataloader

def load_test_data(args):

    # 设置文件夹路径和其他参数
    audio_folder = 'test' # 替换为实际的音频文件夹路径

    # 读取数据并处理
    audio_files = sorted(os.listdir(audio_folder), key=lambda x: int(x[:-4]) if x.endswith('.wav') else float('inf'))

    # 添加路径到文件名
    audio_files_with_path = [os.path.join(audio_folder, file) for file in audio_files]

    # 定义目标采样率
    target_sample_rate = 48000

    # 定义目标长度
    target_length = 256

    # 对音频数据进行重采样、MFCC 转换和归一化
    features = []
    for audio_file in audio_files_with_path:
        print(audio_file)
        # 对音频文件进行去噪处理
        waveform, sample_rate = denoise_audio(audio_file)
        # 没有去噪
        # waveform, sample_rate = torchaudio.load(audio_file)

        # 把音频信号从数组转换为张量，并且转换为浮点类型
        waveform = torch.from_numpy(waveform).float()

        # 定义重采样转换器
        resample = T.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resample(waveform)

        # 定义STFT
        stft = T.Spectrogram(
            n_fft=128,
            win_length=128,
            hop_length=64,
            center=True,
            pad_mode="reflect",
            power=None,
        )

        # 定义 MFCC 转换器
        mfcc = T.MFCC(
            sample_rate=target_sample_rate,
            n_mfcc=40,
            melkwargs={
                'n_fft': 128,
                'n_mels': 128,
                'hop_length': 512,
                # 添加 center 参数为 False，避免在音频信号两端进行填充
                'center': False
            }
        )

        # 计算 MFCC 特征
        feature = mfcc(waveform)
        # 对 MFCC 特征进行归一化
        feature = F.sliding_window_cmn(feature, cmn_window=600, norm_vars=True, center=False)

        # 计算stft，为了方便，生成的变量名不做改变
        # feature = stft(waveform)
        # 把STFT特征转换为幅度谱
        # feature = torch.abs(feature)
        # 对STFT特征进行归一化，并转换为分贝谱
        # feature = F.amplitude_to_DB(feature, multiplier=10.0, amin=1e-10, db_multiplier=0.0)

        # 获取音频文件的长度
        length = feature.shape[2]
        # 如果音频文件的长度小于目标长度，就进行填充
        if length < target_length:
            # 计算需要在两端填充的元素个数
            pad_left = (target_length - length) // 2
            pad_right = target_length - length - pad_left
            # 创建一个填充转换器，指定在两端填充的元素个数和填充值
            pad = torch.nn.ConstantPad1d((pad_left, pad_right), 0.0)
            # 对音频文件进行填充
            feature = pad(feature)
        # 如果音频文件的长度大于目标长度，就进行截断
        elif length > target_length:
            # 对音频文件进行截断，使其长度不超过目标长度
            feature = feature[:, :, :target_length]
        features.append(feature)

    # 对列表中的张量进行转置，使第二维度变为第一维度
    features = [i.transpose(0, 1) for i in features]
    # 使用 pad_sequence 函数填充张量，并指定 batch_first 参数为 True，不指定 total_length 参数
    features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0.0)
    # 对输出张量进行转置，使第二维度变回原来的位置
    features = features.transpose(1, 2)

    test_data_dataloader = DataLoader(features, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 返回 test_data
    return test_data_dataloader
