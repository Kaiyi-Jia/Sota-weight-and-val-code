import os

class Config:
    # -------------------- 数据路径 --------------------
    data_root = "D:/FER_Data"
    dataset = "RAF-DB"

    # -------------------- 模型参数 --------------------
    clip_model_name = "ViT-L/14"
    clip_weights_path = None
    retention_ratio = 0.5
    attn_map_size = (16, 16)
    alpha_hsg = 0.4
    gamma_dgfa = 0.5
    tau_infonce = 0.07
    eps_dar = 1e-6
    w_max_dar = 10.0
    lambda1 = 1.0
    lambda2 = 1.0

    # -------------------- 训练参数 --------------------
    batch_size = 8
    epochs = 100
    lr = 1e-6
    weight_decay = 0.001
    epsilon = 1e-6
    num_workers = 0
    image_size = 224
    device = "cuda"
    seed = 42
    use_amp = True
    eval_epoch_freq = 1
    save_dir = "./checkpoints"
    log_dir = "./logs"

    @property
    def class_num(self):
        if self.dataset == "rafdb":
            return 7
        elif self.dataset == "ferplus":
            return 8
        elif self.dataset == "affectnet7":
            return 7
        elif self.dataset == "affectnet8":
            return 8
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

    @property
    def class_names(self):
        if self.dataset == "rafdb":
            return ["Neutral", "Happy", "Sad", "Surprised", "Scared", "Disgusted", "Angry"]
        elif self.dataset == "ferplus":
            return ["Neutral", "Happy", "Sad", "Surprised", "Scared", "Disgusted", "Angry", "Contemptuous"]
        elif self.dataset == "affectnet7":
            return ["Neutral", "Happy", "Sad", "Surprised", "Scared", "Disgusted", "Angry"]
        elif self.dataset == "affectnet8":
            return ["Neutral", "Happy", "Sad", "Surprised", "Scared", "Disgusted", "Angry", "Contemptuous"]
        else:
            return None
