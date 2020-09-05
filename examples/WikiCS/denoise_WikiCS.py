
from WikiCS import *



MLP_model = MLP(x.size(-1), hidden_channels, dataset.num_classes, device, None, dropout).to(device)

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description="Denoise WikiCS")
    parser.add_argument("--storage", default=osp.join(osp.dirname(osp.realpath(__file__)), ".", "results"),
                        help="Absolute path to store the results of denoising WikiCS")
    parser.add_argument("--split", default=0, help="Which of the 20 splits to use?")
    args = parser.parse_args()
    
    WikiCS(args.storage, args.split)