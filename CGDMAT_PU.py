import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
import torch.nn.init as init
import torch.nn as nn
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from skimage.segmentation import slic,mark_boundaries
import cv2
from cgi import test
from tkinter import Scale
import torchvision.transforms as T
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import torch.optim as optim
from operator import truediv
import time,math
from torch.nn import Module,Conv3d, Parameter,Softmax


def LSC_superpixel(I, nseg):
    superpixelNum = nseg
    ratio = 0.075
    size = int(math.sqrt(((I.shape[0] * I.shape[1]) / nseg)))
    superpixelLSC = cv2.ximgproc.createSuperpixelLSC(
        I,
        region_size=size,
        ratio=0.005)
    superpixelLSC.iterate()
    superpixelLSC.enforceLabelConnectivity(min_element_size=25)
    segments = superpixelLSC.getLabels()
    return np.array(segments,np.int64)

def SEEDS_superpixel(I, nseg):
    I=np.array(I[:, :, 0:3], np.float32).copy()
    I_new = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
    # I_new =np.array( I[:,:,0:3],np.float32).copy()
    height, width, channels = I_new.shape
    superpixelNum = nseg
    seeds = cv2.ximgproc.createSuperpixelSEEDS(width, height, channels, int(superpixelNum), num_levels=2,prior=1,histogram_bins=5)
    seeds.iterate(I_new,4)
    segments = seeds.getLabels()
    # segments=SegmentsLabelProcess(segments) # 排除labels中不连续的情况
    return segments

def SegmentsLabelProcess(labels):
    '''
    对labels做后处理，防止出现label不连续现象
    '''
    labels = np.array(labels, np.int64)
    H, W = labels.shape
    ls = list(set(np.reshape(labels, [-1]).tolist()))
    
    dic = {}
    for i in range(len(ls)):
        dic[ls[i]] = i
    
    new_labels = labels
    for i in range(H):
        for j in range(W):
            new_labels[i, j] = dic[new_labels[i, j]]
    return new_labels

class SLIC(object):
    def __init__(self, HSI,labels, n_segments=1000, compactness=20, max_iter=20, sigma=0, min_size_factor=0.3,
                 max_size_factor=2):
        self.n_segments = n_segments
        self.compactness = compactness
        self.max_iter = max_iter
        self.min_size_factor = min_size_factor
        self.max_size_factor = max_size_factor
        self.sigma = sigma
        # 数据standardization标准化,即提前全局BN
        height, width, bands = HSI.shape  # 原始高光谱数据的三个维度
        data = np.reshape(HSI, [height * width, bands])
        minMax = preprocessing.StandardScaler()
        data = minMax.fit_transform(data)
        self.data = np.reshape(data, [height, width, bands])
        self.labels=labels
        
    
    def get_Q_and_S_and_Segments(self):
        # 执行 SLCI 并得到Q(nxm),S(m*b)
        img = self.data
        (h, w, d) = img.shape
        # 计算超像素S以及相关系数矩阵Q
        segments = slic(img, n_segments=self.n_segments, compactness=self.compactness, max_iter=self.max_iter,
                        convert2lab=False,sigma=self.sigma, enforce_connectivity=True,
                        min_size_factor=self.min_size_factor, max_size_factor=self.max_size_factor,slic_zero=False)
        
        # 判断超像素label是否连续,否则予以校正
        if segments.max()+1!=len(list(set(np.reshape(segments,[-1]).tolist()))): segments = SegmentsLabelProcess(segments)
        self.segments = segments
        superpixel_count = segments.max() + 1
        self.superpixel_count = superpixel_count
        print("superpixel_count", superpixel_count)
        
        # ######################################显示超像素图片
        out = mark_boundaries(img[:,:,[0,1,2]], segments)
        # out = (img[:, :, [0, 1, 2]]-np.min(img[:, :, [0, 1, 2]]))/(np.max(img[:, :, [0, 1, 2]])-np.min(img[:, :, [0, 1, 2]]))
        '''plt.figure()
        plt.imshow(out)
        plt.show()'''
        
        segments = np.reshape(segments, [-1])
        S = np.zeros([superpixel_count, d], dtype=np.float32)
        Q = np.zeros([w * h, superpixel_count], dtype=np.float32)
        x = np.reshape(img, [-1, d])
        
        for i in range(superpixel_count):
            idx = np.where(segments == i)[0]
            count = len(idx)
            pixels = x[idx]
            superpixel = np.sum(pixels, 0) / count
            S[i] = superpixel
            Q[idx, i] = 1
        
        self.S = S
        self.Q = Q
        
        return Q, S , self.segments
    
    def get_A(self, sigma: float):
        '''
         根据 segments 判定邻接矩阵
        :return:
        '''
        A = np.zeros([self.superpixel_count, self.superpixel_count], dtype=np.float32)
        (h, w) = self.segments.shape
        for i in range(h - 2):
            for j in range(w - 2):
                sub = self.segments[i:i + 2, j:j + 2]
                sub_max = np.max(sub).astype(np.int32)
                sub_min = np.min(sub).astype(np.int32)
                # if len(sub_set)>1:
                if sub_max != sub_min:
                    idx1 = sub_max
                    idx2 = sub_min
                    if A[idx1, idx2] != 0:
                        continue
                    
                    pix1 = self.S[idx1]
                    pix2 = self.S[idx2]
                    diss = np.exp(-np.sum(np.square(pix1 - pix2)) / sigma ** 2)
                    A[idx1, idx2] = A[idx2, idx1] = diss
        
        return A

class LDA_SLIC(object):
    def __init__(self,data,labels,n_component):
        self.data=data
        self.init_labels=labels
        self.curr_data=data
        self.n_component=n_component
        self.height,self.width,self.bands=data.shape
        self.x_flatt=np.reshape(data,[self.width*self.height,self.bands])
        self.y_flatt=np.reshape(labels,[self.height*self.width])
        self.labes=labels
        
    def LDA_Process(self,curr_labels):
        '''
        :param curr_labels: height * width
        :return:
        '''
        curr_labels=np.reshape(curr_labels,[-1])
        idx=np.where(curr_labels!=0)[0]
        x=self.x_flatt[idx]
        y=curr_labels[idx]
        lda = LinearDiscriminantAnalysis()#n_components=self.n_component
        lda.fit(x,y-1)
        X_new = lda.transform(self.x_flatt)
        return np.reshape(X_new,[self.height, self.width,-1])
       
    def SLIC_Process(self,img,scale=25):
        n_segments_init=self.height*self.width/scale
        print("n_segments_init",n_segments_init)
        myslic=SLIC(img,n_segments=n_segments_init,labels=self.labes, compactness=1,sigma=1, min_size_factor=0.1, max_size_factor=2)
        Q, S, Segments = myslic.get_Q_and_S_and_Segments()
        A=myslic.get_A(sigma=10)
        return Q,S,A,Segments
        
    def simple_superpixel(self,scale):
        curr_labels = self.init_labels
        X = self.LDA_Process(curr_labels)
        Q, S, A, Seg = self.SLIC_Process(X,scale=scale)
        return Q, S, A,Seg
    
    def simple_superpixel_no_LDA(self,scale):
        Q, S, A, Seg = self.SLIC_Process(self.data,scale=scale)
        return Q, S, A,Seg

MIN_NUM_PATCHES = 16
class GCNLayer(nn.Module):
    def __init__(self, input_dim:int, output_dim:int,A:torch.Tensor):
        super(GCNLayer, self).__init__()
        self.A = A
        self.BN = nn.BatchNorm1d(input_dim)
        self.Activition = nn.LeakyReLU()
        self.sigmal = torch.nn.Parameter(torch.tensor([0.1],requires_grad=True))
        #第一层网络
        self.GCN_liner_theta_1 = nn.Sequential(nn.Linear(input_dim,256))
        self.GCN_liner_out_1 = nn.Sequential(nn.Linear(input_dim,output_dim))
        nodes_count = self.A.shape[0]
        self.I = torch.eye(nodes_count,nodes_count,requires_grad=False)
        self.mask = torch.ceil(self.A * 0.00001)

    def A_to_D_inv(self,A:torch.Tensor):
        D = A.sum(1)
        D_hat = torch.diag(torch.pow(D,-0.5))
        return D_hat
    
    def forward(self,H,model='normal'):
        #采用softmax归一化实现加速运算
        H = self.BN(H)
        H_xxl = self.GCN_liner_theta_1(H)
        e = torch.sigmoid(torch.matmul(H_xxl,H_xxl.t()))
        zero_vec = -9e15 * torch.ones_like(e)
        A = torch.where(self.mask>0,e,zero_vec)+self.I
        if model !='normal':# This is a trick for the Indian Pines
            A = torch.clamp(A,0.1)
        A = F.softmax(A,dim=1)
        output = self.Activition(torch.mm(A,self.GCN_liner_out_1(H)))
        return output,A

class CEGCN(nn.Module):
    def __init__(self, height:int, width:int, channel:int, class_count:int,
                 Q:torch.Tensor, A:torch.Tensor, model='normal'):
        super(CEGCN, self).__init__()
        self.class_count = class_count
        self.channel = channel
        self.height = height
        self.width = width
        self.Q = Q
        self.A = A
        self.model = model
        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))#列归一化Q
        layers_count = 2
        # Spectra Transformation Sub-NetWork
        self.CNN_denoise = nn.Sequential()
        for i in range(layers_count):
            if i == 0:
                self.CNN_denoise.add_module('CNN_denoise_BN'+str(i),nn.BatchNorm2d(self.channel))
                self.CNN_denoise.add_module('CNN_denoise_Conv'+str(i),nn.Conv2d(self.channel, 128, kernel_size=(1,1)))
                self.CNN_denoise.add_module('CNN_denoise_Act'+str(i),nn.LeakyReLU())
            else:
                self.CNN_denoise.add_module('CNN_denoise_BN'+str(i),nn.BatchNorm2d(128))
                self.CNN_denoise.add_module('CNN_denoise_Conv'+str(i),nn.Conv2d(128, 128, kernel_size=(1,1)))
                self.CNN_denoise.add_module('CNN_denoise_Act'+str(i),nn.LeakyReLU())
        # Superpixel-level Graph Sub-NetWork
        self.GCN_Branch = nn.Sequential()
        for i in range(layers_count):
            if i<layers_count-1:
                self.GCN_Branch.add_module('GCN_Bratch'+str(i),GCNLayer(128, 128, self.A))
            else:
                self.GCN_Branch.add_module('GCN_Bratch'+str(i),GCNLayer(128, 64, self.A))
        # Softmax Layer
        self.Softmax_linear = nn.Sequential(nn.Linear(128, self.class_count))
    
    def forward(self, x:torch.Tensor):
        # x: H*W*C return probability_map
        (h,w,c) = x.shape
        # 先去除噪声
        #print("x.shape:",x.shape)
        #print("torch.unsqueeze(x.permute([0,2,1]),0):",torch.unsqueeze(x.permute([0,2,1]),0).size())
        noise = self.CNN_denoise(torch.unsqueeze(x.permute([2,0,1]),0))
        #print("noise.shape:",noise.shape)
        noise = torch.squeeze(noise,0).permute([1,2,0])
        clean_x = noise #直连
        clean_x_flatten = clean_x.reshape([h*w,-1])
        superpixels_flatten = torch.mm(self.norm_col_Q.t(), clean_x_flatten) # 低频部分
        hx = clean_x
        # GCN层1 转化为超像素 x_flat 乘以 列归一化Q
        H = superpixels_flatten
        if self.model == 'normal':
            for i in range(len(self.GCN_Branch)):
                H, _ = self.GCN_Branch[i](H)
        else:
            for i in range(len(self.GCN_Branch)):
                H, _ = self.GCN_Branch[i](H,model='smoothed')
        GCN_result = torch.matmul(self.Q, H)
        #两组特征融合
        Y = GCN_result
        '''
        Y = self.Softmax_linear(Y)
        Y = F.softmax(Y,-1)
        '''
        return Y

samples_type = ['ratio', 'same_num'][0] # FLAG 1:Indian 2:PaviaU 3: Salinas
curr_train_ratio = 0.1
Scale = 100
data_mat = sio.loadmat('../input/hybridsn/data/PaviaU.mat')
data = data_mat['paviaU']
gt_mat = sio.loadmat('../input/hybridsn/data/PaviaU_gt.mat')
gt = gt_mat['paviaU_gt']
class_count = 9
learning_rate = 5e-4
superpixel_scale = Scale
train_samples_per_class = curr_train_ratio
val_samples =class_count
train_ratio = curr_train_ratio
cmap = cm.get_cmap('jet', class_count+1)
plt.set_cmap(cmap)
m, n, d = data.shape
# 提前全局BN
orig_data = data
height, width, bands = data.shape
data = np.reshape(data,[height*width, bands])
minMax = preprocessing.StandardScaler()
data = minMax.fit_transform(data)
data = np.reshape(data, [height,width,bands])
gt_reshape = np.reshape(gt,[-1])
net_input=np.array(data,np.float32)
net_input=torch.from_numpy(net_input.astype(np.float32))
print("LDA-SLIC Operation is Processing ")
ls = LDA_SLIC(data,np.reshape(gt_reshape,[height,width]),class_count-1)
Q, S ,A,Seg= ls.simple_superpixel(scale=superpixel_scale)
Q=torch.from_numpy(Q)
A=torch.from_numpy(A)
GCNmodel = CEGCN(height, width, bands, class_count, Q, A, model='smoothed')
print("parameters:\n", GCNmodel.parameters(), len(list(GCNmodel.parameters())))
optimizer = torch.optim.Adam(GCNmodel.parameters(),lr=learning_rate)#,weight_decay=0.0001
best_loss=99999
GCNmodel.train()
output = GCNmodel(net_input)
output = output.reshape([-1,height,width])
transform = T.ToTensor()
#data = transform(data)
#Newdata = torch.cat([data,output],dim=0)
#Newdata = Newdata.permute([1,2,0]).detach().numpy()#np.transpose(Newdata,(1,2,0))
#print("output shape :",output.shape) # torch.Size([64, 145, 145])
#print("orig_data shape:",orig_data.shape) # orig_data shape: (145, 145, 200)
#print("Newdata shape:",Newdata.shape) # Newdata shape: (145, 145, 264)

def infoChange(X,numComponents):
    X_copy = np.zeros((X.shape[0] , X.shape[1], X.shape[2]))
    half = int(numComponents/2)
    for i in range(0,half-1):
        X_copy[:,:,i] = X[:,:,(half-i)*2-1]
    for i in range(half,numComponents):
        X_copy[:,:,i] = X[:,:,(i-half)*2]
    X = X_copy
    return X

# 对高光谱数据 X 应用 PCA 变换
def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX,(X.shape[0], X.shape[1], numComponents))
    newX = infoChange(newX,numComponents)
    return newX

# 对单个像素周围提取 patch 时，边缘像素就无法取了，因此，给这部分像素进行 padding 操作
def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

# 在每个像素周围提取 patch ，然后创建成符合 keras 处理的格式
def createImageCubes(X, y, windowSize=5, removeZeroLabels = True):
    # 给 X 做 padding
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    return patchesData, patchesLabels

def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=testRatio,random_state=randomState,stratify=y)
    return X_train, X_test, y_train, y_test

BATCH_SIZE_TRAIN = 64

def create_data_loader():
    # 地物类别
    # class_num = 9
    # 读入数据
    X, y = data,gt
    # 用于测试样本的比例
    test_ratio = 0.9
    # 每个像素周围提取 patch 的尺寸
    patch_size = 13
    # 使用 PCA 降维，得到主成分的数量
    pca_components = 100
    print('Hyperspectral data shape: ', X.shape)
    print('Label shape: ', y.shape)
    print('\n... ... PCA tranformation ... ...')
    X_pca = applyPCA(X, numComponents=pca_components)
    #HSI先做PCA和通道平移策略，再与GCN进行concate操作
    X_pca = transform(X_pca)
    X_pca = torch.cat([X_pca,output],dim=0)
    X_pca = X_pca.permute([1,2,0]).detach().numpy()
    print('Data shape after PCA: ', X_pca.shape)
    print('\n... ... create data cubes ... ...')
    X_pca, y = createImageCubes(X_pca, y, windowSize=patch_size)
    print('Data cube X shape: ', X_pca.shape)
    print('Data cube y shape: ', y.shape)
    print('\n... ... create train &np.squeeze test data ... ...')
    Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X_pca, y, test_ratio)
    print('Xtrain shape: ', Xtrain.shape)
    print('Xtest  shape: ', Xtest.shape)
    # 改变 Xtrain, Ytrain 的形状，以符合 keras 的要求
    X = X_pca.reshape(-1, patch_size, patch_size, pca_components+output.shape[0], 1)
    Xtrain = Xtrain.reshape(-1, patch_size, patch_size, pca_components+output.shape[0], 1)
    Xtest = Xtest.reshape(-1, patch_size, patch_size, pca_components+output.shape[0], 1)
    print('before transpose: Xtrain shape: ', Xtrain.shape)
    print('before transpose: Xtest  shape: ', Xtest.shape)
    # 为了适应 pytorch 结构，数据要做 transpose
    X = X.transpose(0, 4, 3, 1, 2)
    Xtrain = Xtrain.transpose(0, 4, 3, 1, 2)
    Xtest = Xtest.transpose(0, 4, 3, 1, 2)
    print('after transpose: Xtrain shape: ', Xtrain.shape)
    print('after transpose: Xtest  shape: ', Xtest.shape)
    # 创建train_loader和 test_loader
    X = TestDS(X, y)
    trainset = TrainDS(Xtrain, ytrain)
    testset = TestDS(Xtest, ytest)
    train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                               batch_size=BATCH_SIZE_TRAIN,
                                               shuffle=True,
                                               drop_last=True
                                               )
    test_loader = torch.utils.data.DataLoader(dataset=testset,
                                               batch_size=BATCH_SIZE_TRAIN,
                                               shuffle=False,
                                               num_workers=0,
                                               drop_last=True
                                              )
    all_data_loader = torch.utils.data.DataLoader(dataset=X,
                                                batch_size=BATCH_SIZE_TRAIN,
                                                shuffle=False,
                                                num_workers=0,
                                                drop_last=True
                                              )

    return train_loader, test_loader, all_data_loader, y

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# 等于 PreNorm
class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# 等于 FeedForward
class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim),nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim,Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(LayerNormalize(dim,MLP_Block(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x

NUM_CLASS = 9

class PAM_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width, channle = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height*channle).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height*channle)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height*channle)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width, channle)
        # print('out', out.shape)
        # print('x', x.shape)
        out = self.gamma*out + x
        #print('out', out.shape)
        return out

class CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width, channle = x.size()
        #print(x.size())
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1) #形状转换并交换维度
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width, channle)
        # print('out', out.shape)
        # print('x', x.shape)
        out = self.gamma*out + x  #C*H*W
        #print('out', out.shape)
        return out

class ViT(nn.Module):
    def __init__(self, *,in_channels=1,image_size, patch_size, num_classes, dim, depth, heads,
                 mlp_dim, channels = 64, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert num_patches > NUM_CLASS, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'
        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(174,dim)#patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        self.attention_spectral = CAM_Module(174)
        self.attention_spatial = PAM_Module(64)

        self.conv2d_f = nn.Sequential(
            nn.Conv2d(in_channels=32*164,out_channels=64,kernel_size=(3,3),padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv3d_f = nn.Sequential(
            nn.Conv3d(in_channels,out_channels=32,kernel_size=(3,3,3),padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
        )
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels, out_channels=32, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
        )
        self.conv3d_features_1 = nn.Sequential(
            nn.Conv3d(in_channels=33, out_channels=64, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
        )

        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=64 * 164, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv2d_features_1 = nn.Sequential(
            nn.Conv2d(in_channels=64*164+64, out_channels=110, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(110),
            nn.ReLU(),
        )
       
    def forward(self, img, mask = None):
        p = self.patch_size
        #分支一
        #卷积+串行
        x1 = self.conv3d_f(img)
        x1 = self.attention_spectral(x1)
        x1 = rearrange(x1,'b c l h w -> b (l c) h w')
        x1 = self.conv2d_f(x1)
        x1 = (torch.unsqueeze(x1,0)).permute([1,2,0,3,4])
        x1 = self.attention_spatial(x1)
        x1 = rearrange(x1,'b c l h w -> b (c l) h w')# x1 shape: torch.Size([64, 64, 1, 13, 13])
        #分支二
        res = img
        x2 = self.conv3d_features(img)#64
        x2 = torch.cat((x2, res), dim=1)
        x2 = self.conv3d_features_1(x2)
        res1 = rearrange(x2, 'b c h w y -> b (c h) w y')#64
        x2 = x2.reshape(x2.shape[0], x2.shape[1] * x2.shape[2], x2.shape[3], x2.shape[4])
        x2 = self.conv2d_features(x2)
        x2 = torch.cat((x2, res1), dim=1)
        x2 = self.conv2d_features_1(x2)
        img = torch.cat([x1,x2],dim=1)
        #print("img shape：",img.shape)#img shape： torch.Size([64, 1, 174, 13, 13])
        #img = img.reshape(img.shape[0],img.shape[1]*img.shape[2],img.shape[3],img.shape[4])
        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        #print("before x:",x.size())
        x = self.patch_to_embedding(x)
        #print("after x:",x.size())
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x, mask)
        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)

""" Training dataset"""
class TrainDS(torch.utils.data.Dataset):
    def __init__(self, Xtrain, ytrain):
        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)
    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        # 返回文件数据的数目
        return self.len

""" Testing dataset"""
class TestDS(torch.utils.data.Dataset):
    def __init__(self, Xtest, ytest):
        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)
    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        # 返回文件数据的数目
        return self.len

def train(train_loader, epochs):
    # 使用GPU训练，可以在菜单 "代码执行工具" -> "更改运行时类型" 里进行设置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 网络放到GPU上
    #Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
    #image_size, patch_size, num_classes, dim, depth, heads,mlp_dim
    #patch_dim = channels * patch_size ** 2
    net = ViT(image_size = 13,patch_size = 1,num_classes = NUM_CLASS,dim = 1024,depth = 2,
              heads = 16,mlp_dim = 2048,channels =164,dropout = 0.1,emb_dropout = 0.1).to(device)
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 初始化优化器
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    # 开始训练
    total_loss = 0
    for epoch in range(epochs):
        net.train()
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            # 正向传播 +　反向传播 + 优化
            # 通过输入得到预测的输出
            outputs = net(data)
            #print("outputs:",outputs.size())
            # 计算损失函数
            loss = criterion(outputs, target)
            # 优化器梯度归零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]' % (epoch + 1,
                                                                         total_loss / (epoch + 1),
                                                                         loss.item()))
    print('Finished Training')
    return net, device

def test(device, net, test_loader):
    count = 0
    # 模型测试
    net.eval()
    y_pred_test = 0
    y_test = 0
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = net(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = outputs
            y_test = labels
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, labels))

    return y_pred_test, y_test

def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def acc_reports(y_test, y_pred_test):
    target_names = ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets', 'Bare Soil', 'Bitumen',
                    'Self-Blocking Bricks', 'Shadows']
    classification = classification_report(y_test, y_pred_test, digits=4, target_names=target_names)
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)

    return classification, oa*100, confusion, each_acc*100, aa*100, kappa*100

if __name__ == '__main__':
    train_loader, test_loader, all_data_loader, y_all= create_data_loader()
    tic1 = time.perf_counter()
    net, device = train(train_loader, epochs=400)
    # 只保存模型参数
    torch.save(net.state_dict(),'PU_net_params.pth')
    toc1 = time.perf_counter()
    tic2 = time.perf_counter()
    y_pred_test, y_test = test(device, net, test_loader)
    toc2 = time.perf_counter()
    # 评价指标
    classification, oa, confusion, each_acc, aa, kappa = acc_reports(y_test, y_pred_test)
    classification = str(classification)
    Training_Time = toc1 - tic1
    Test_time = toc2 - tic2
    localtime = time.strftime('%Y-%m-%d %H:%M',time.localtime(time.time()))
    file_name = "./HSI_classification_PU_report_"+str(localtime)+".txt"
    with open(file_name, 'w') as x_file:
        x_file.write('{} Training_Time (s)'.format(Training_Time))
        x_file.write('\n')
        x_file.write('{} Test_time (s)'.format(Test_time))
        x_file.write('\n')
        x_file.write('{} Kappa accuracy (%)'.format(kappa))
        x_file.write('\n')
        x_file.write('{} Overall accuracy (%)'.format(oa))
        x_file.write('\n')
        x_file.write('{} Average accuracy (%)'.format(aa))
        x_file.write('\n')
        x_file.write('{} Each accuracy (%)'.format(each_acc))
        x_file.write('\n')
        x_file.write('{}'.format(classification))
        x_file.write('\n')
        x_file.write('{}'.format(confusion))
