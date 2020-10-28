from model_processor import Processor
import argparse

def _init(parser):
    parser.add_argument('-is_train', type=int, action='store')
    parser.add_argument('-fold', type=int, nargs='+', action='store')
    parser.add_argument('-batch_size', type=int, action='store')
    parser.add_argument('-stop_num', type=int, action='store')
    parser.add_argument('-num_epoches', type=int, action='store')
    parser.add_argument('-learning_rate', type=float, nargs='?', action='store', default=5e-5)
    parser.add_argument('-pretrained_model', nargs='?', action='store', default='bert-base-chinese')
    parser.add_argument('-trained_models', nargs='?', action='store', default=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process hyperparameters')
    _init(parser)
    args = parser.parse_args()

    # init core processor
    processor = Processor(args)
    
    # run processor
    processor.run()