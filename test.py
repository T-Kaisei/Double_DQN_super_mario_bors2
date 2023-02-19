from network import Qnet


def check_gpu():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('GPU is available')
    else:
        device = torch.device("cpu")
        print('GPU is not available')


if __name__ == '__main__':
    check_gpu()