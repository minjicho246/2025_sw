import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


import os
from torch.utils.data import Dataset
from monai.transforms import Compose, LoadImaged, EnsureTyped, Resized, NormalizeIntensityd, ToTensord,RandFlipd,SqueezeDimd
# 추가적으로 test와 train transform 예시 (필요 시 RandFlipd, MinMaxIntensity 등의 transform 추가)

# Training transform (2D 이미지의 경우)
train_transforms = Compose([
    LoadImaged(keys=["image", "label"], ensure_channel_first=True),
    EnsureTyped(keys=["image", "label"]),
    Resized(keys=["image", "label"], spatial_size=(128, 128)),  # (H, W, D) 크기를 (128,128,1)로 조정
    # SqueezeDimd(keys=["image", "label"], dim=-1),                 # 결과: (1,128,128)
    NormalizeIntensityd(keys=["image"]),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),  
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),  
    ToTensord(keys=["image", "label"]),
])

# test transform: 랜덤 flip 없이 resize와 intensity normalization만 적용
test_transforms = Compose([
    LoadImaged(keys=["image", "label"], ensure_channel_first=True),
    EnsureTyped(keys=["image", "label"]),
    Resized(keys=["image", "label"], spatial_size=(128, 128)),
    # SqueezeDimd(keys=["image", "label"], dim=-1),                 # 결과: (1,128,128)

    NormalizeIntensityd(keys=["image"]),
    ToTensord(keys=["image", "label"]),
])


class NSCLCDataset(Dataset):
    def __init__(self, df, transform=None):
        """
        df: pandas DataFrame, 첫 번째 컬럼은 PatientID,
            'Patient_Summary', 'deadstatus.event', 'Survival.time' 등의 컬럼을 포함해야 함.
        transform: MONAI Compose 객체 (예, train_transforms 또는 test_transforms)
        """
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 환자 ID 가져오기 (예: 'LUNG1-128' 등)
        patient_id = self.df.iloc[idx]['PatientID']
        # print(f"Processing Patient ID: {patient_id}")
        # 환자 폴더에 저장된 이미지와 마스크 파일 경로 구성
        # 예: '/workspace/2025.05_정보통신학회/Study/Data/{PatientID}_img.nii.gz'
        img_file = os.path.join(img_path, f'{patient_id}_img.nii')
        mask_file = os.path.join(img_path, f'{patient_id}_mask.nii')
        
        # MONAI transform을 적용하기 위한 dictionary 구성
        data = {
            "image": img_file, 
            "label": mask_file,
            "text": str(self.df.iloc[idx]['Patient_Summary']),
            "event": int(self.df.iloc[idx]['deadstatus.event']),
            "time": float(self.df.iloc[idx]['Survival.time'])
        }
        
        if self.transform:
            random_seed = np.random.randint(0, 10000)
            data = self.transform(data)
        
        # output은 기존과 동일하게 반환 (MONAI transform 적용 후 image와 label은 Tensor로 변환되어 있음)
        return data

# 사용 예시 (training 데이터)
dataset = NSCLCDataset(df, transform=train_transforms)
data = dataset[0]
print(data["image"].shape, data["label"].shape)
from sentence_transformers import SentenceTransformer

# from transformers import AutoTokenizer, AutoModel
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
class SentenceTransformersEmbedding(nn.Module):
    def __init__(self, model_name="all-MiniLM-L6-v2", embedding_dim=128, device=None, dropout_rate=0.3):
        """
        sentence-transformers를 사용한 텍스트 임베딩 클래스 (sentence-transformers 모델 freeze).
        
        Args:
            model_name (str): sentence-transformers 모델 이름 (기본: "all-MiniLM-L6-v2")
            embedding_dim (int): 출력 임베딩 차원 (기본: 128)
            device (torch.device): 사용할 디바이스 (None이면 자동 감지)
            dropout_rate (float): Dropout 비율 (기본: 0.3)
        """
        super(SentenceTransformersEmbedding, self).__init__()

        # Device 설정 (자동 감지)
        self.device = device #if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # sentence-transformers 모델 로드 (GPU 자동 할당)
        self.sentence_transformer = SentenceTransformer(model_name).to(self.device)

        # sentence-transformers 모델 freeze (gradient 업데이트 방지)
        for param in self.sentence_transformer.parameters():
            param.requires_grad = False

        # 모델 출력 차원 확인 (all-MiniLM-L6-v2는 384차원)
        input_dim = self.sentence_transformer.get_sentence_embedding_dimension()

        # MLP 레이어 (임베딩 차원 축소, 학습 가능)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),#512->
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),  # Dropout 추가 (30% 확률)
            nn.Linear(256, embedding_dim)
        )

    def forward(self, text_batch):
        """
        텍스트 배치를 입력받아 임베딩 생성.
        
        Args:
            text_batch (list or str): 텍스트 리스트 또는 단일 텍스트 문자열
        
        Returns:
            torch.Tensor: 축소된 임베딩 (batch_size, embedding_dim)
        """
        # # 텍스트가 단일 문자열이면 리스트로 변환
        # if isinstance(text_batch, str):
        #     text_batch = [text_batch]

        # sentence-transformers로 임베딩 생성
        embeddings = self.sentence_transformer.encode(text_batch, convert_to_tensor=True, device=self.device)
        # 출력 크기: (batch_size, input_dim)

        # MLP를 통해 차원 축소
        reduced_output = self.mlp(embeddings)

        return reduced_output
    
import torch
import torch.nn as nn

class SurvivalMLP(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, output_dim=1):
        super(SurvivalMLP, self).__init__()
        self.mlp = nn.Sequential(
            # 1st layer
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            # 2nd layer
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # 3rd layer
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # 4th layer
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # 5th (output) layer
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)
class CrossModalContrastiveLossWithMemoryBankTensor(nn.Module):
    def __init__(self, temperature=0.07, memory_size=800, embedding_dim=128):
        super().__init__()
        self.temperature = temperature
        self.memory_size = memory_size
        self.embedding_dim = embedding_dim

        # 메모리 뱅크를 텐서로 관리 (초기에는 비어있음)
        self.register_buffer('text_memory', torch.empty(0, embedding_dim))
        self.register_buffer('image_memory', torch.empty(0, embedding_dim))
    def reset_memory_bank(self):
        self.text_memory = torch.empty(0, self.embedding_dim, device=self.text_memory.device)
        self.image_memory = torch.empty(0, self.embedding_dim, device=self.image_memory.device)
        print("Memory Bank Reset!")
    def update_memory_bank(self, new_embeds, memory):
        """
        memory: [current_mem_size, embedding_dim]
        new_embeds: [batch_size, embedding_dim]
        """
        # 과거 임베딩은 detach() 상태로 유지(gradient 불필요)
        # 하지만 현재 배치(new_embeds)는 detach 안 하면, '이전 배치'와 그래프 충돌 발생 가능
        # => 보통은 과거 임베딩만 detach하고, 현재 배치는 그대로 쓰되,
        #    메모리 뱅크에 들어가는 시점에 detach()해서 '다음 배치'에서만 참조하도록 함.

        # 1) concat
        updated_bank = torch.cat([memory, new_embeds.detach()], dim=0)  
        #    └ detach() → "다음 배치에서"만 쓰기 위해 그래프 분리

        # 2) FIFO 유지
        if updated_bank.size(0) > self.memory_size:
            updated_bank = updated_bank[-self.memory_size:]

        return updated_bank
    def forward(self, text_embeddings, image_embeddings):
        # (A) 현재 배치 정규화 (gradient 필요)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)  # [batch_size, embedding_dim]
        image_embeddings = F.normalize(image_embeddings, p=2, dim=1)  # [batch_size, embedding_dim]

        # (B) 현재 메모리 뱅크 가져오기 (업데이트 전 과거 데이터만)
        text_mem_past = self.text_memory  # [N_prev, embedding_dim]
        image_mem_past = self.image_memory  # [N_prev, embedding_dim]

        # (C) 현재 배치와 과거 메모리 결합
        text_all = torch.cat([text_mem_past, text_embeddings], dim=0)  # [N_prev + batch_size, embedding_dim]
        image_all = torch.cat([image_mem_past, image_embeddings], dim=0)  # [N_prev + batch_size, embedding_dim]

        # (D) 유사도 계산: 전체 vs 전체
        logits_text_to_image = torch.matmul(image_all, text_all.T) / self.temperature  # [N, N]
        logits_image_to_text = torch.matmul(text_all, image_all.T) / self.temperature  # [N, N]

        # (E) Positive 레이블: 같은 인덱스만
        N = text_all.size(0)
        labels = torch.arange(N, device=text_embeddings.device)

        # (F) Contrastive 손실
        loss_text_to_image = F.cross_entropy(logits_text_to_image, labels)
        loss_image_to_text = F.cross_entropy(logits_image_to_text, labels)
        loss = (loss_text_to_image + loss_image_to_text) / 2.0

        # (G) 메모리 뱅크 업데이트 (손실 계산 후, gradient 영향을 주지 않음)
        self.text_memory = self.update_memory_bank(text_embeddings, self.text_memory)
        self.image_memory = self.update_memory_bank(image_embeddings, self.image_memory)

        return loss



# class SupervisedContrastiveLoss(nn.Module):
#     def __init__(self, gamma=1.0, bank_size=1024, feature_dim=128):
#         super(SupervisedContrastiveLoss, self).__init__()
#         self.gamma = gamma
#         self.bank_size = bank_size
#         self.feature_dim = feature_dim
#         self.register_buffer('memory_features', torch.zeros(bank_size, feature_dim))
#         self.register_buffer('memory_times', torch.zeros(bank_size))
#         self.register_buffer('memory_events', torch.zeros(bank_size))
#         self.bank_ptr = 0
#         self.bank_full = False

#     def reset_memory_bank(self):
#         self.memory_features.zero_()
#         self.memory_times.zero_()
#         self.memory_events.zero_()
#         self.bank_ptr = 0
#         self.bank_full = False

#     @staticmethod
#     def cosine_similarity(vec1, vec2):
#         return F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0))

#     def rbf_kernel(self, t_i, t_k):
#         return torch.exp(-self.gamma * (t_i - t_k) ** 2)

#     def update_memory_bank(self, t, e, features):
#         batch_size = features.size(0)
#         if self.bank_ptr + batch_size > self.bank_size:
#             remain = self.bank_size - self.bank_ptr
#             self.memory_features[self.bank_ptr:self.bank_size] = features[:remain]
#             self.memory_times[self.bank_ptr:self.bank_size] = t[:remain]
#             self.memory_events[self.bank_ptr:self.bank_size] = e[:remain]
#             self.memory_features[0:batch_size - remain] = features[remain:]
#             self.memory_times[0:batch_size - remain] = t[remain:]
#             self.memory_events[0:batch_size - remain] = e[remain:]
#             self.bank_ptr = batch_size - remain
#             self.bank_full = True
#         else:
#             self.memory_features[self.bank_ptr:self.bank_ptr + batch_size] = features
#             self.memory_times[self.bank_ptr:self.bank_ptr + batch_size] = t
#             self.memory_events[self.bank_ptr:self.bank_ptr + batch_size] = e
#             self.bank_ptr += batch_size

#     def forward(self, t, e, features):
#         t_detached = t.detach()
#         e_detached = e.detach()
#         features_detached = features.detach()
#         epsilon = 1e-8
#         if self.bank_full:
#             all_features = torch.cat([features, self.memory_features], dim=0)
#             all_t = torch.cat([t, self.memory_times], dim=0)
#             all_e = torch.cat([e, self.memory_events], dim=0)
#         else:
#             all_features = features
#             all_t = t
#             all_e = e
#         N = t.shape[0]
#         total_loss = torch.zeros(1, device=t.device, requires_grad=True)
#         self.update_memory_bank(t_detached, e_detached, features_detached)
#         return total_loss / N if N > 0 else torch.zeros(1, device=t.device, requires_grad=True)


class NegativeLogLikelihood(nn.Module):
    def __init__(self, l2_reg=None):
        super().__init__()
        self.is_reg = l2_reg is not None
        self.reg = Regularization(order=2, weight_decay=l2_reg) if self.is_reg else None

    def forward(self, risk_pred, censor, y, device, model=None):
        e = censor
        mask = torch.ones(y.shape[0], y.shape[0]).to(device, dtype=torch.float)
        mask[(y.T - y) > 0] = 0
        log_loss = torch.exp(risk_pred) * mask
        log_loss = torch.sum(log_loss, dim=0) / torch.sum(mask, dim=0)
        log_loss = torch.log(log_loss).reshape(-1, 1)
        neg_log_loss = -torch.sum((risk_pred - log_loss) * e) / (torch.sum(e) + 1e-6)
        if self.is_reg:
            neg_log_loss += self.reg(model)
        return neg_log_loss

class Regularization(object):
    def __init__(self, order, weight_decay):
        self.order = order
        self.weight_decay = weight_decay

    def __call__(self, model):
        reg_loss = sum(torch.norm(w, p=self.order) for name, w in model.named_parameters())
        return self.weight_decay * reg_loss
class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, gamma=1.0, bank_size=1024, feature_dim=128, eps=1e-8):
        super().__init__()
        self.gamma = gamma
        self.bank_size = bank_size
        self.feature_dim = feature_dim
        self.eps = eps

        self.register_buffer('memory_features', torch.zeros(bank_size, feature_dim))
        self.register_buffer('memory_times',    torch.zeros(bank_size))
        self.register_buffer('memory_events',   torch.zeros(bank_size))
        self.bank_ptr = 0
        self.bank_full = False

    def reset_memory_bank(self):
        self.memory_features.zero_()
        self.memory_times.zero_()
        self.memory_events.zero_()
        self.bank_ptr = 0
        self.bank_full = False

    def update_memory_bank(self, t, e, features):
        B = features.size(0)
        if self.bank_ptr + B > self.bank_size:
            wrap = (self.bank_ptr + B) - self.bank_size
            cut = self.bank_size - self.bank_ptr
            # 뒤에서부터 채우고 남으면 앞쪽에 덮어쓰기
            self.memory_features[self.bank_ptr:] = features[:cut]
            self.memory_times[self.bank_ptr:]    = t[:cut]
            self.memory_events[self.bank_ptr:]   = e[:cut]
            self.memory_features[:wrap] = features[cut:]
            self.memory_times[:wrap]    = t[cut:]
            self.memory_events[:wrap]   = e[cut:]
            self.bank_ptr = wrap
            self.bank_full = True
        else:
            self.memory_features[self.bank_ptr:self.bank_ptr+B] = features
            self.memory_times[self.bank_ptr:self.bank_ptr+B]    = t
            self.memory_events[self.bank_ptr:self.bank_ptr+B]   = e
            self.bank_ptr += B

    def forward(self, t, e, features):
        """
        t: (B,)   survival times
        e: (B,)   event flags (0=censored, 1=uncensored)
        features: (B,D) embeddings
        """
        device = features.device
        B, D = features.shape

        # detach 해서 memory 용으로만 사용
        t_det = t.detach()
        e_det = e.detach()
        f_det = features.detach()

        # 현재 배치 + memory bank 결합
        if self.bank_full:
            all_feats = torch.cat([features, self.memory_features], dim=0)  # (M,D)
            all_t     = torch.cat([t, self.memory_times], dim=0)           # (M,)
            all_e     = torch.cat([e, self.memory_events], dim=0)          # (M,)
        else:
            all_feats = features
            all_t     = t
            all_e     = e

        M = all_feats.size(0)
        total_loss = 0.0

        for i in range(B):
            # Anchor는 반드시 uncensored여야 함
            if e[i] != 1:
                continue

            anchor_feat = features[i]
            anchor_t    = t[i]

            # Positive: uncensored 전체 중 i 제외
            pos_mask = (all_e == 1)
            # anchor 위치(i)만 features 쪽에 존재하므로, same index만 제외
            pos_mask[i] = False
            pos_idx = pos_mask.nonzero(as_tuple=False).view(-1)
            if pos_idx.numel() == 0:
                continue

            # Negative: censored(t_event=0) & t_k > t_i
            neg_mask = (all_e == 0) & (all_t > anchor_t)
            neg_idx = neg_mask.nonzero(as_tuple=False).view(-1)
            if neg_idx.numel() == 0:
                continue

            # 긍정군 유사도 & 가중치
            sims_pos = F.cosine_similarity(anchor_feat.unsqueeze(0),
                                           all_feats[pos_idx], dim=1)    # (P,)
            weights_pos = torch.exp(-self.gamma * (anchor_t - all_t[pos_idx])**2)
            num = (torch.exp(sims_pos) * weights_pos).sum() + self.eps

            # 부정군 유사도 & 가중치
            sims_neg = F.cosine_similarity(anchor_feat.unsqueeze(0),
                                           all_feats[neg_idx], dim=1)    # (N,)
            weights_neg = torch.exp(-self.gamma * (anchor_t - all_t[neg_idx])**2)
            # (1 - w_k)을 similarity에 곱해줌
            neg_term = torch.exp(sims_neg * (1.0 - weights_neg)).sum()

            denom = num + neg_term + self.eps

            loss_i = -torch.log(num / denom)
            total_loss += loss_i

        # memory bank 업데이트
        self.update_memory_bank(t_det, e_det, f_det)

        # 평균 loss
        return total_loss / B
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, groups=groups, bias=False, dilation=dilation
    )

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2   = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class ResNet18Embedding(nn.Module):
    def __init__(self, in_channels=3, embed_dim=128):
        super().__init__()
        # Initial conv + maxpool
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Layers
        self.layer1 = self._make_layer(64,  64,  blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128,  blocks=2, stride=2)
        self.layer3 = self._make_layer(128,256,  blocks=2, stride=2)
        self.layer4 = self._make_layer(256,512,  blocks=2, stride=2)

        # Adaptive pool + embedding
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, embed_dim)

    def _make_layer(self, in_planes, planes, blocks, stride):
        downsample = None
        if stride != 1 or in_planes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes * BasicBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * BasicBlock.expansion),
            )
        layers = [BasicBlock(in_planes, planes, stride, downsample)]
        in_planes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Input x: (B, C, 128, 128)
        x = self.conv1(x)   # (B,64,64,64)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # (B,64,32,32)

        x = self.layer1(x)  # (B,64,32,32)
        x = self.layer2(x)  # (B,128,16,16)
        x = self.layer3(x)  # (B,256,8,8)
        x = self.layer4(x)  # (B,512,4,4)

        x = self.avgpool(x) # (B,512,1,1)
        x = torch.flatten(x, 1)  # (B,512)
        embedding = self.fc(x)   # (B,128)
        return embedding

# Example usage:
if __name__ == "__main__":
    model = ResNet18Embedding(in_channels=1, embed_dim=128)  # for grayscale use in_channels=1
    dummy = torch.randn(4, 1, 128, 128)
    out = model(dummy)
    print(out.shape)  # should be torch.Size([4, 128])
def train_step1_code(imgmodel, textmodel, device,
               learning_rate_step1=1e-5, 
               num_epochs=num_epochs, batch_size=batch_size,
               clinical_pd=df,  # 전체 임상 DataFrame, 반드시 'PatientID', 'Patient_Summary', 'deadstatus.event', 'Survival.time' 등이 포함
               dataset_folder=img_path,   # 이미지와 마스크 파일이 저장된 폴더 (파일명은 "{PatientID}_img.nii.gz" 와 "{PatientID}_mask.nii.gz")
               output_folder=None):
    # 모델을 device로 이동
    # unet.to(device)
    imgmodel.to(device)
    textmodel.to(device)
    # Optimizer 구성
    optimizer_step1 = torch.optim.Adam(
        list(imgmodel.parameters()) + list(textmodel.parameters()), lr=learning_rate_step1
    )

    # 손실 함수 정의
    crossmodal_loss = CrossModalContrastiveLossWithMemoryBankTensor().to(device)
    # mseloss = nn.MSELoss().to(device)
    img_risk_loss = SupervisedContrastiveLoss().to(device)
    negative_risk_loss = NegativeLogLikelihood().to(device)
    # 데이터셋 구성
    # 데이터 split (8:1:1)
    train_pd, temp_pd = train_test_split(clinical_pd, test_size=0.4, stratify=clinical_pd['deadstatus.event'], random_state=random_seed)
    valid_pd, test_pd = train_test_split(temp_pd, test_size=0.5, stratify=temp_pd['deadstatus.event'], random_state=random_seed)
    train_pd.reset_index(drop=True, inplace=True)
    valid_pd.reset_index(drop=True, inplace=True)
    test_pd.reset_index(drop=True, inplace=True)
    # 각 데이터셋의 데이터셋 객체 생성 (여기서는 NSCLCDataset 사용)
    train_dataset = NSCLCDataset(train_pd, train_transforms)
    valid_dataset = NSCLCDataset(valid_pd, test_transforms)
    test_dataset  = NSCLCDataset(test_pd, test_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    # 모든 Epoch의 결과를 저장할 리스트
    training_results = []
    # Step 6: Training loop
    step1_best_loss = float('inf')
    step2_best_loss = float('inf')
    step3_best_loss = float('inf')
    scaler = GradScaler()  # AMP 손실 스케일러
    
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        # Step 1: Train the image and text models
        imgmodel.train()
        textmodel.train()
        total_loss_step1 = 0
        crossmodal_loss.reset_memory_bank()  # 메모리 뱅크 초기화
        for batch in train_loader:
            img = batch['image'].to(device)
            text = batch['text']
            optimizer_step1.zero_grad()
            with autocast():
                img_embedding = imgmodel(img)
                text_embedding = textmodel(text)
                loss_crossmodal = crossmodal_loss(text_embedding, img_embedding)
                total_loss = loss_crossmodal
            scaler.scale(total_loss).backward()
            scaler.step(optimizer_step1)
            scaler.update()
            total_loss_step1 += total_loss.item()
        total_loss_step1 /= len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss 1: {total_loss_step1:.4f}")
        
        #valid
        imgmodel.eval()
        textmodel.eval()
        crossmodal_loss.reset_memory_bank()  # 메모리 뱅크 초기화
        with torch.no_grad():
            valid_loss = 0
            for batch in valid_loader:
                img = batch['image'].to(device)
                text = batch['text']
                img_embedding = imgmodel(img)
                text_embedding = textmodel(text)
                loss_crossmodal = crossmodal_loss(text_embedding, img_embedding)
                valid_loss += loss_crossmodal.item()
            valid_loss /= len(valid_loader)
            print(f"Validation Loss: {valid_loss:.4f}")
        # Save the best model
        if valid_loss < step1_best_loss:
            step1_best_loss = valid_loss
            save_path = os.path.join(output_folder, f"best_step1_{epoch+1}.pth")
            torch.save({
                # 'unet_state_dict': unet.state_dict()
                'imgmodel_state_dict': imgmodel.state_dict(),
                'textmodel_state_dict': textmodel.state_dict(),
                'optimizer_state_dict': optimizer_step1.state_dict(),
                'best_valid_loss': step1_best_loss,
                'epoch': epoch+1
            }, save_path)
            print(f"Best model updated at epoch {epoch+1} (Valid Loss: {step1_best_loss:.4f}), saved to {save_path}")
            
            #Test
            imgmodel.eval()
            textmodel.eval()
            crossmodal_loss.reset_memory_bank()  # 메모리 뱅크 초기화
            with torch.no_grad():
                test_loss = 0
                for batch in test_loader:
                    img = batch['image'].to(device)
                    text = batch['text']
                    img_embedding = imgmodel(img)
                    text_embedding = textmodel(text)
                    loss_crossmodal = crossmodal_loss(text_embedding, img_embedding)
                    test_loss += loss_crossmodal.item()
                test_loss /= len(test_loader)
                print(f"Test Loss: {test_loss:.4f}")
            training_results.append({
                'Epoch': epoch + 1,
                'Train_Loss': total_loss_step1,
                'Valid_Loss': valid_loss,
                'Test_Loss': test_loss
            })
    # Save training results to pkl
    with open(os.path.join(output_folder, 'step1_results.pkl'), 'wb') as f:
        pickle.dump(training_results, f)
    print(f"Training results saved to {os.path.join(output_folder, 'step1_results.pkl')}")
    return training_results

    def train_step2_code(imgmodel, device,
               learning_rate_step2=1e-5,
               num_epochs=num_epochs, batch_size=batch_size,
               clinical_pd=df,  # 전체 임상 DataFrame, 반드시 'PatientID', 'Patient_Summary', 'deadstatus.event', 'Survival.time' 등이 포함
               dataset_folder=img_path,   # 이미지와 마스크 파일이 저장된 폴더 (파일명은 "{PatientID}_img.nii.gz" 와 "{PatientID}_mask.nii.gz")
               output_folder=None):
    # 모델을 device로 이동
    # unet.to(device)
    
    imgmodel.to(device)
    #load checkpoint
    checkpoint = torch.load(os.path.join(output_folder, 'best_step1_59.pth'), map_location=device)
    imgmodel.load_state_dict(checkpoint['imgmodel_state_dict'])
    # textmodel.to(device)
    # Optimizer 구성
    optimizer_step2 = torch.optim.Adam(
        list(imgmodel.parameters()), lr=learning_rate_step2
    )
    

    # 손실 함수 정의
    # crossmodal_loss = CrossModalContrastiveLossWithMemoryBankTensor().to(device)
    # mseloss = nn.MSELoss().to(device)
    img_risk_loss = SupervisedContrastiveLoss().to(device)
    # 데이터셋 구성
    # 데이터 split (8:1:1)
    train_pd, temp_pd = train_test_split(clinical_pd, test_size=0.4, stratify=clinical_pd['deadstatus.event'], random_state=random_seed)
    valid_pd, test_pd = train_test_split(temp_pd, test_size=0.5, stratify=temp_pd['deadstatus.event'], random_state=random_seed)
    train_pd.reset_index(drop=True, inplace=True)
    valid_pd.reset_index(drop=True, inplace=True)
    test_pd.reset_index(drop=True, inplace=True)
    # 각 데이터셋의 데이터셋 객체 생성 (여기서는 NSCLCDataset 사용)
    train_dataset = NSCLCDataset(train_pd, train_transforms)
    valid_dataset = NSCLCDataset(valid_pd, test_transforms)
    test_dataset  = NSCLCDataset(test_pd, test_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    # 모든 Epoch의 결과를 저장할 리스트
    training_results = []
    # Step 6: Training loop
    step2_best_loss = float('inf')
    scaler = GradScaler()  # AMP 손실 스케일러
    
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        # Step 1: Train the image and text models
        imgmodel.train()
        total_loss_step2 = 0
        # crossmodal_loss.reset_memory_bank()  # 메모리 뱅크 초기화
        img_risk_loss.reset_memory_bank()  # 메모리 뱅크 초기화
        for batch in train_loader:
            img = batch['image'].to(device)
            # time = batch['time'].to(device)
            # event = batch['event'].to(device)
            optimizer_step2.zero_grad()
            img_embedding = imgmodel(img)
            total_loss = img_risk_loss(batch['time'].to(device), batch['event'].to(device), img_embedding)
            # with autocast():
            #     img_embedding = imgmodel(img)
                
            #     loss_supervised = img_risk_loss(batch['time'].to(device), batch['event'].to(device), img_embedding)
            #     total_loss = loss_supervised
            # scaler.scale(total_loss).backward()
            # scaler.step(optimizer_step2)
            # scaler.update()
            total_loss_step2 += total_loss.item()
        total_loss_step2 /= len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss 1: {total_loss_step2:.4f}")
        
        #valid
        imgmodel.eval()
        # crossmodal_loss.reset_memory_bank()  # 메모리 뱅크 초기화
        img_risk_loss.reset_memory_bank()  # 메모리 뱅크 초기화
        with torch.no_grad():
            valid_loss = 0
            for batch in valid_loader:
                img = batch['image'].to(device)
                img_embedding = imgmodel(img)
                # valid_loss += loss_crossmodal.item()
                valid_loss += img_risk_loss(batch['time'].to(device), batch['event'].to(device), img_embedding).item()
            valid_loss /= len(valid_loader)
            print(f"Validation Loss: {valid_loss:.4f}")
        # Save the best model
        if valid_loss < step2_best_loss:
            step2_best_loss = valid_loss
            save_path = os.path.join(output_folder, f"best_step2_{epoch+1}.pth")
            torch.save({
                # 'unet_state_dict': unet.state_dict()
                'imgmodel_state_dict': imgmodel.state_dict(),
                'optimizer_state_dict': optimizer_step2.state_dict(),
                'best_valid_loss': step2_best_loss,
                'epoch': epoch+1
            }, save_path)
            print(f"Best model updated at epoch {epoch+1} (Valid Loss: {step2_best_loss:.4f}), saved to {save_path}")
            
            #Test
            imgmodel.eval()
            img_risk_loss.reset_memory_bank()  # 메모리 뱅크 초기화
            with torch.no_grad():
                test_loss = 0
                for batch in test_loader:
                    img = batch['image'].to(device)
                    img_embedding = imgmodel(img)
                    test_loss += img_risk_loss(batch['time'].to(device), batch['event'].to(device), img_embedding).item()
                test_loss /= len(test_loader)
                print(f"Test Loss: {test_loss:.4f}")
            training_results.append({
                'Epoch': epoch + 1,
                'Train_Loss': total_loss_step2,
                'Valid_Loss': valid_loss,
                'Test_Loss': test_loss
            })
    # Save training results to pkl
    with open(os.path.join(output_folder, 'step2_results.pkl'), 'wb') as f:
        pickle.dump(training_results, f)
    print(f"Training results saved to {os.path.join(output_folder, 'step2_results.pkl')}")
    return training_results

            
def train_step3_code(imgmodel,mlpmodel, device,
               learning_rate_step3=1e-5,
               num_epochs=num_epochs, batch_size=batch_size,
               clinical_pd=df,  # 전체 임상 DataFrame, 반드시 'PatientID', 'Patient_Summary', 'deadstatus.event', 'Survival.time' 등이 포함
               dataset_folder=img_path,   # 이미지와 마스크 파일이 저장된 폴더 (파일명은 "{PatientID}_img.nii.gz" 와 "{PatientID}_mask.nii.gz")
               output_folder=None):
    # 모델을 device로 이동
    # unet.to(device)
    
    imgmodel.to(device)
    mlpmodel.to(device)
    #load checkpoint
    checkpoint = torch.load(os.path.join(output_folder, 'best_step2_51.pth'), map_location=device)
    imgmodel.load_state_dict(checkpoint['imgmodel_state_dict'])
    # textmodel.to(device)
    # Optimizer 구성
    optimizer_step3 = torch.optim.Adam(
        list(imgmodel.parameters())+list(mlpmodel.parameters()), lr=learning_rate_step3
    )
    

    # 손실 함수 정의
    surv_loss = NegativeLogLikelihood().to(device)
    # 데이터셋 구성
    # 데이터 split (8:1:1)
    train_pd, temp_pd = train_test_split(clinical_pd, test_size=0.4, stratify=clinical_pd['deadstatus.event'], random_state=random_seed)
    valid_pd, test_pd = train_test_split(temp_pd, test_size=0.5, stratify=temp_pd['deadstatus.event'], random_state=random_seed)
    train_pd.reset_index(drop=True, inplace=True)
    valid_pd.reset_index(drop=True, inplace=True)
    test_pd.reset_index(drop=True, inplace=True)
    # 각 데이터셋의 데이터셋 객체 생성 (여기서는 NSCLCDataset 사용)
    train_dataset = NSCLCDataset(train_pd, train_transforms)
    valid_dataset = NSCLCDataset(valid_pd, test_transforms)
    test_dataset  = NSCLCDataset(test_pd, test_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    # 모든 Epoch의 결과를 저장할 리스트
    training_results = []
    # Step 6: Training loop
    step3_best_loss = float('inf')
    scaler = GradScaler()  # AMP 손실 스케일러
    
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        # Step 1: Train the image and text models
        imgmodel.train()
        mlpmodel.train()
        total_loss_step3 = 0
        y_true, y_pred, event_list = [], [], []
        # crossmodal_loss.reset_memory_bank()  # 메모리 뱅크 초기화
        for batch in train_loader:
            img = batch['image'].to(device)
            event = batch['event'].to(device)
            time = batch['time'].to(device)
        
            optimizer_step3.zero_grad()
            img_embedding = imgmodel(img)
            risk_pred = mlpmodel(img_embedding)
            risk_pred.reshape(-1,1)
            total_loss = surv_loss(risk_pred, event, time, device)
            
            total_loss_step3 += total_loss.item()
            time.reshape(-1,1)
            event.reshape(-1,1)
            y_true.extend(time.cpu().numpy())
            y_pred.extend(risk_pred.cpu().detach().numpy())
            event_list.extend(event.cpu().numpy())
        total_loss_step3 /= len(train_loader)
       
        train_df = pd.DataFrame({
            'risk_score': [float(r) if isinstance(r, (int, float)) else float(r[0]) for r in y_pred],
            'time': y_true,
            'event': event_list
        })
        # print(train_df)
        try:
            train_c_index = concordance_index(train_df['time'], -train_df['risk_score'], train_df['event'])
        except Exception as e:
            print("Error calculating C-Index", e)
            train_c_index = 0
        median_risk = train_df['risk_score'].median()
        train_low_df = train_df[train_df['risk_score']< median_risk]
        train_high_df = train_df[train_df['risk_score']>= median_risk]
        train_log = logrank_test(train_low_df['time'], train_high_df['time'], train_low_df['event'], train_high_df['event'])
        print(f"Epoch {epoch+1}/{num_epochs}, Loss 1: {total_loss_step3:.4f}, Train C-Index: {train_c_index:.4f}, Train LogRank: {train_log.p_value:.4f}")
        
        #valid
        imgmodel.eval()
        mlpmodel.eval()
        # crossmodal_loss.reset_memory_bank()  # 메모리 뱅크 초기화
        # img_risk_loss.reset_memory_bank()  # 메모리 뱅크 초기화
        
        with torch.no_grad():
            valid_loss = 0
            y_true, y_pred, event_list = [], [], []
            for batch in valid_loader:
                img = batch['image'].to(device)
                event = batch['event'].to(device)
                time = batch['time'].to(device)
                img_embedding = imgmodel(img)
                risk_pred = mlpmodel(img_embedding)
                risk_pred.reshape(-1,1)
                valid_loss += surv_loss(risk_pred, event, time, device).item()
                event.reshape(-1,1)
                time.reshape(-1,1)
                y_true.extend(time.cpu().numpy())
                y_pred.extend(risk_pred.cpu().detach().numpy())
                event_list.extend(event.cpu().numpy())
                
            valid_loss /= len(valid_loader)
            
            valid_df = pd.DataFrame({
                'risk_score': [float(r) if isinstance(r, (int, float)) else float(r[0]) for r in y_pred],
                'time': y_true,
                'event': event_list
            })
            try:
                valid_c_index = concordance_index(valid_df['time'], -valid_df['risk_score'], valid_df['event'])
            except:
                valid_c_index = 0
                
            # median_risk = valid_df['risk_score'].median()
            valid_low_df = valid_df[valid_df['risk_score']< median_risk]
            valid_high_df = valid_df[valid_df['risk_score']>= median_risk]
            valid_log = logrank_test(valid_low_df['time'], valid_high_df['time'], valid_low_df['event'], valid_high_df['event'])
            print(f"Validation Loss: {valid_loss:.4f}, Valid C-Index: {valid_c_index:.4f}, Valid LogRank: {valid_log.p_value:.4f}")
        # Save the best model
        if valid_loss < step3_best_loss:
            step3_best_loss = valid_loss
            save_path = os.path.join(output_folder, f"best_step3_{epoch+1}.pth")
            torch.save({
                # 'unet_state_dict': unet.state_dict()
                'imgmodel_state_dict': imgmodel.state_dict(),
                'mlpmodel_state_dict': mlpmodel.state_dict(),
                'optimizer_state_dict': optimizer_step3.state_dict(),
                'best_valid_loss': step3_best_loss,
                'epoch': epoch+1
            }, save_path)
            print(f"Best model updated at epoch {epoch+1} (Valid Loss: {step3_best_loss:.4f}), saved to {save_path}")
            
            #Test
        imgmodel.eval()
        mlpmodel.eval()
        # img_risk_loss.reset_memory_bank()  # 메모리 뱅크 초기화
        with torch.no_grad():
            test_loss = 0
            y_true, y_pred, event_list = [], [], []
            for batch in test_loader:
                img = batch['image'].to(device)
                event = batch['event'].to(device)
                time = batch['time'].to(device)
                img_embedding = imgmodel(img)
                risk_pred = mlpmodel(img_embedding)
                risk_pred.reshape(-1,1)
                test_loss += surv_loss(risk_pred, event, time, device).item()
                event.reshape(-1,1)
                time.reshape(-1,1)
                y_true.extend(time.cpu().numpy())
                y_pred.extend(risk_pred.cpu().detach().numpy())
                event_list.extend(event.cpu().numpy())
            
            test_df = pd.DataFrame({
                'risk_score': [float(r) if isinstance(r, (int, float)) else float(r[0]) for r in y_pred],
                'time': y_true,
                'event': event_list
            })
            try:
                test_c_index = concordance_index(test_df['time'], -test_df['risk_score'], test_df['event'])
            except:
                test_c_index = 0
                
            # median_risk = test_df['risk_score'].median()
            test_low_df = test_df[test_df['risk_score']< median_risk]
            test_high_df = test_df[test_df['risk_score']>= median_risk]
            test_log = logrank_test(test_low_df['time'], test_high_df['time'], test_low_df['event'], test_high_df['event'])
            
            test_loss /= len(test_loader)
            print(f"Test Loss: {test_loss:.4f}, Test C-Index: {test_c_index:.4f}, Test LogRank: {test_log.p_value:.4f}")
            training_results.append({
                'Epoch': epoch + 1,
                'Train_Loss': total_loss_step3,
                'Valid_Loss': valid_loss,
                'Test_Loss': test_loss,
                'Train_C_Index': train_c_index,
                'Valid_C_Index': valid_c_index,
                'Test_C_Index': test_c_index,
                'Train_LogRank': train_log.p_value,
                'Valid_LogRank': valid_log.p_value,
                'Test_LogRank': test_log.p_value
            })
    # Save training results to pkl
    with open(os.path.join(output_folder, 'step3_results.pkl'), 'wb') as f:
        pickle.dump(training_results, f)
    print(f"Training results saved to {os.path.join(output_folder, 'step3_results.pkl')}")
    return training_results

            
            
        
        
        
        
        
    

        
        
        
        
        
    
      
            
        
        
        
        
        
    
