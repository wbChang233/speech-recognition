import argparse
import warnings
import srnet as srnet
from data_processor import *
from trainer import *

warnings.filterwarnings("ignore")


def main():
    global args
    args = parser.parse_args()
    print(args)

    if args.model == 'srnet':
        model = srnet.SRNet()
        args.epochs = 60
        args.save_dir = 'trained_srnet'
        dic = args.save_dir + f"/{args.model}_{args.epochs}.pth"

    model = model.cuda()

    model.load_state_dict(torch.load(dic))

    test_data_loader = load_test_data(args)
    model.eval()

    # 设定一个阈值，比如0.5
    threshold = 0.65

    predictions_list = []
    with torch.no_grad():
        for inputs in tqdm(test_data_loader):
            inputs = inputs.to('cuda')  # 将输入数据转移到cuda设备上
            outputs = model(inputs)
            # 使用softmax函数将输出转换为概率分布
            probs = torch.softmax(outputs, dim=1)
            # 找出概率最大的类别和对应的概率值
            Predictions_top1 = probs.argmax(dim=1)
            max_probs = probs.max(dim=1)[0]
            # 判断概率值是否大于阈值，如果小于阈值，则将类别标记为-1
            Predictions_top1[max_probs < threshold] = -1
            predictions_list.append(Predictions_top1.tolist())

    # 得到结果
    char_to_int = {'C': 0, 'M': 1, 'E': 2}
    # Use a dictionary comprehension to reverse the mapping
    int_to_char = {v: k for k, v in char_to_int.items()}
    # 将-1类别对应的字符设为0
    int_to_char[-1] = '0'
    # Use the join method to concatenate the characters
    predictions_str = ''.join([int_to_char[pred] for preds in predictions_list for pred in preds])

    print(predictions_str)

    # 打开一个文件对象
    text_file = open("result/predictions.txt", "w")
    # 把字符串写入文件
    text_file.write(predictions_str)
    # 关闭文件
    text_file.close()


if __name__ == '__main__':
    torch.cuda.set_device(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='srnet',
                        choices=['srnet'],
                        help='choose model to train')
    parser.add_argument('--epochs', default=60, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--save-dir', dest='save_dir',
                        help='The directory used to save the trained model',
                        default='trained_srnet', type=str)

    main()
