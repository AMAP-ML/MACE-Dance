from args import parse_train_opt
from EDGE import EDGE


def test(opt):
    checkpoint_path = './runs/train/exp/weights/train-3750.pt'
    model = EDGE(opt.feature_type, checkpoint_path=checkpoint_path)
    model.test_loop(opt)


if __name__ == "__main__":
    opt = parse_train_opt()
    test(opt)
