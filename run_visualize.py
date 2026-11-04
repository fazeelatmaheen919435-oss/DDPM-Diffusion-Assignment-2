import sys
sys.path.insert(0, "/kaggle/working/Assignment_2")
from utils import visualize_diffusion
visualize_diffusion(train_steps=int(sys.argv[1]), save_and_sample_every=int(sys.argv[2]), fid=False,
    save_folder="./results/q6_4", data_path=sys.argv[3], load_path="./results/q6_3/model.pt",
    time_steps=50, dataloader_workers=2)
