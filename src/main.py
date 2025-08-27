import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader,random_split
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
import numpy
import matplotlib.pyplot as plt
from PIL import Image as IMG
import xml.etree.ElementTree as ET
import os
from torch.optim import Adam
from tqdm import tqdm

class MyDataset(Dataset):
    def __init__(self,root_dir,train=True,transform=None):
        super(MyDataset,self).__init__()
        self.root_dir = root_dir
        self.img_dir = "JPEGImages"
        self.annot_dir = "Annotations"
        self.transform = transform
        self.train = train
        self.img_root=os.path.join(root_dir,self.img_dir)
        self.annot_root=os.path.join(root_dir,self.annot_dir)
        self.img_files =os.listdir(self.img_root)
        self.annot_files =os.listdir(self.annot_root)

        self.name_to_num={
            'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4,
            'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9,
            'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14,
            'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19
        }
    def parse_xml(self,annot_file):
        tree=ET.parse(annot_file)
        root=tree.getroot()
        boxes=[]
        labels=[]
        for obj in root.findall('object'):
            clas=self.name_to_num[obj.find('name').text]
            bnbox=obj.find('bndbox')
            xmin=float(bnbox.find('xmin').text)
            xmax=float(bnbox.find('xmax').text)
            ymin=float(bnbox.find('ymin').text)
            ymax=float(bnbox.find('ymax').text)

            boxes.append([xmin,ymin,xmax,ymax])
            labels.append(clas)

        return torch.tensor(boxes,dtype=torch.float32),torch.tensor(labels,dtype=torch.float32)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):

        img_root=os.path.join(self.root_dir,self.img_dir)
        annot_root=os.path.join(self.root_dir,self.annot_dir)
        img_path=os.path.join(self.img_root,self.img_files[index])
        annot_path=os.path.join(self.annot_root,self.annot_files[index])
        #print(img_path)

        img=IMG.open(img_path).convert('RGB')
        totensor=transforms.ToTensor()

        img=totensor(img)
        boxes,labels=self.parse_xml(annot_path)

        if self.train and  self.transform is not None:#数据集够大，之后再加涉及对boxes进行更改的数据增强
            boxes_=[]
            target_size=(604,604)
            c,h,w=img.shape
            h_t,w_t=target_size

            resize=transforms.Resize(target_size)
            img=resize(img)
            img=self.transform(img)
            scale_x=w_t/w; scale_y=h_t/h

            for box in boxes:
                xmin, ymin, xmax, ymax = box
                xmin=xmin*scale_x; ymin=ymin*scale_y
                xmax=xmax*scale_x; ymax=ymax*scale_y
                box=torch.tensor([xmin,ymin,xmax,ymax],dtype=torch.float32)
                boxes_.append(box)

            boxes=boxes_

        target={
            'boxes':boxes,
            'labels':labels,
            'id':torch.tensor([index]),
        }
        return img,target

def devide(img):
     b,c,w,h=img.shape
     w_=w if w%2==0 else w+1
     h_=h if h%2==0 else h+1
     img=F.interpolate(img,size=(w_,h_),mode='bilinear')
     img1=img[...,...,::2,1::2]
     img2=img[...,...,1::2,1::2]
     img3=img[...,...,1::2,::2]
     img4=img[...,...,::2,::2]
     return torch.cat([img1,img2,img3,img4],dim=1)#在通道道上进行拼接，在第一层获得更大的感受野，同时等效于用四个卷积核去捕捉一种局部特征，增强提取能力

def swish(x):
    return x * torch.sigmoid(x)

def feat_up(x1,x2):

    shape1,shape2=x1.shape,x2.shape
    h_=shape1[2] if shape1[2]>shape2[2] else shape2[2]
    w_=shape1[3] if shape1[3]>shape2[3] else shape2[3]

    x1=F.interpolate(x1,size=(h_,w_),mode='nearest')
    x2=F.interpolate(x2,size=(h_,w_),mode='nearest')

    return x1,x2

class CovActBn(nn.Module):

    def __init__(self,cin,cout,k_size=3,pad=1,stride=1):

        super(CovActBn, self).__init__()
        self.conv=nn.Conv2d(cin,cout,k_size,stride=stride,padding=pad)
        self.bn=nn.BatchNorm2d(cout)

    def forward(self,x):

        return swish(self.bn(self.conv(x)))

class Elan(nn.Module):

    def __init__(self,cin,cout):

        super(Elan,self).__init__()
        self.cov1=CovActBn(cin,cin,1,1)
        self.cov2=CovActBn(cin,cin,1,1)
        self.cov3=CovActBn(cin,cin,3,1)
        self.cov4=CovActBn(cin,cin,3,1)
        self.cov5=CovActBn(cin,cin,3,1)
        self.cov6=CovActBn(cin,cin,3,1)
        self.cov7=CovActBn(4*cin,cout,1,1)

    def forward(self,x):

        x1=self.cov1(x)
        x2=self.cov2(x)
        x3=self.cov4(self.cov3(x2))
        x4=self.cov6(self.cov5(x3))
        y=self.cov7(torch.cat((x1,x2,x3,x4),dim=1))

        return y

class Elan_cat(nn.Module):

    def __init__(self,cin,cout):

        super(Elan_cat,self).__init__()
        self.cov_=CovActBn(cin,cin,1,1)
        self.cov1=CovActBn(cin,cin,1,1)
        self.cov2=CovActBn(cin,cin,1,1)
        self.cov3=CovActBn(cin,cin,3,1)
        self.cov4=CovActBn(cin,cin,3,1)
        self.cov5=CovActBn(cin,cin,3,1)
        self.cov6=CovActBn(cin,cin,3,1)
        self.cov7=CovActBn(5*cin,cout,1,1)

    def forward(self,x,x_):

        x,x_=feat_up(x,x_)
        x_=self.cov_(x_)
        x1=self.cov1(x)
        x2=self.cov2(x)
        x3=self.cov4(self.cov3(x2))
        x4=self.cov6(self.cov5(x3))
        y=self.cov7(torch.cat([x1,x2,x3,x4,x_],dim=1))

        return y

class Wide_Elan(nn.Module):

    def __init__(self,cin,cout):

        super(Wide_Elan,self).__init__()
        self.cov1=CovActBn(cin,cin,1,1)
        self.cov2=CovActBn(cin,cin,1,1)
        self.cov3=CovActBn(cin,cin,3,1)
        self.cov4=CovActBn(cin,cin,3,1)
        self.cov5=CovActBn(cin,cin,3,1)
        self.cov6=CovActBn(cin,cin,3,1)
        self.cov7=CovActBn(6*cin,cout,1,1)

    def forward(self,x):
        x1=self.cov1(x)
        x2=self.cov2(x)
        x3=self.cov3(x2)
        x4=self.cov4(x3)
        x5=self.cov5(x4)
        x6=self.cov6(x5)
        y=self.cov7(torch.cat([x1,x2,x3,x4,x5,x6],dim=1))

        return y

class multyCov (nn.Module):

    def __init__(self,cin,cout,nl):

        super(multyCov,self).__init__()
        self.net=nn.ModuleList([])

        for i in range(nl-1):
            self.net.append(CovActBn(cin,cin,3,1))
        self.net.append(CovActBn(cin,cout,1,1))

    def forward(self,x):
        y=x
        for layer in self.net:
            y=layer(y)
        return y

class SPPCSPC(nn.Module):

    def __init__(self,cin,cout,e=0.5,k=(5,9,13)):

        super(SPPCSPC,self).__init__()
        c_=int(2*cin*e)

        self.cv1=CovActBn(cin,c_,1,0)
        self.cv2=CovActBn(c_,c_,1,0)
        self.SPPF=nn.ModuleList( [ nn.MaxPool2d(kernel_size=x,padding=x//2,stride=1) for x in k ] ) #使用并行池化来快速增大感受野
        self.cv3=CovActBn(len(k)*c_+c_,c_,1,0)
        self.cv4=CovActBn(2*c_,cout,1,0)

    def forward(self,x):

        x1=self.cv1(x)
        x2=self.cv2(x1)
        pools=[x2]
        for maxpool in self.SPPF:
            pool=maxpool(x2)
            pools.append(pool)
        x3=self.cv3(torch.cat(pools,dim=1))
        x3=torch.cat([x1,x3],dim=1)
        x4=self.cv4(x3)

        return x4

class UpBlock(nn.Module):

    def __init__(self,cin,cout):

        super(UpBlock,self).__init__()
        self.cov1=CovActBn(cin,cout,3,1)
        self.up=nn.ConvTranspose2d(cout,cout,2,2,1,bias=False)

    def forward(self,x):

        return self.up(self.cov1(x))

class MaxPool_cat(nn.Module):

    def __init__(self,cin):

        super(MaxPool_cat,self).__init__()
        self.cov=CovActBn(cin,2*cin,1,1)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2,padding=1)

    def forward(self,x,x_):

        x=self.pool(self.cov(x))
        x,x_=feat_up(x,x_)
        y=torch.cat([x,x_],dim=1)

        return y

class detect(nn.Module):

    def __init__(self,cin,cout,k,p,na,nc):
        super(detect,self).__init__()
        self.na=na
        self.nc=nc
        self.cov=CovActBn(cin,cout,k,p)

    def forward(self,x):
        y=self.cov(x)
        b,c,h,w=y.shape
        y=y.reshape( b , self.na , (5+self.nc) , h, w ).permute( 0 , 1 , 3 , 4 ,  2 )

        return y

class net(nn.Module):

    def __init__(self,class_num,anchor_num):

        super(net,self).__init__()

        self.nc=class_num
        self.na=anchor_num
        out_ch = self.na * (self.nc + 5)
        self.cov1 = CovActBn(3,32,3,1)
        self.cov2 = CovActBn(32,64,3,1,stride=2) #1/2
        self.cov3 = CovActBn(64,128,3,1)
        self.elan1 = Elan(128,256)  #1/4
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.elan2 = Elan(256,512)#1/8
        self.pool2 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.elan3 = Elan(512, 1024)#1/16
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.elan4 = Elan(1024, 1024)#1/32
        self.neck =SPPCSPC(1024,512)
        self.concatp3 = CovActBn(1024,256,1,0)
        self.concatp2 = CovActBn(512,128,1,0)
        self.up1 = UpBlock(512,256)
        self.up2 = UpBlock(256,128)
        self.elan_1=Elan_cat(256,256)
        self.elan_2=Elan_cat(128,128)
        self.pool4=MaxPool_cat(128)#p3
        self.pool5=MaxPool_cat(256)#p4
        self.elan_3=Wide_Elan(512,256)
        self.elan_4=Wide_Elan(1024,512)
        self.out_cov1=CovActBn(128,256,3,1)
        self.out_cov2=CovActBn(256,512,3,1)
        self.out_cov3=CovActBn(512,1024,3,1)

        self.detect1=detect(256,out_ch,1,1,self.na,self.nc)
        self.detect2=detect(512,out_ch,1,1,self.na,self.nc)
        self.detect3=detect(1024,out_ch,1,1,self.na,self.nc)

    def forward(self,x):
        x1=self.cov1(x)
        x2=self.cov2(x1) #p1
        x3=self.cov3(x2)
        x4=self.elan1(x3)
        x5=self.pool1(x4) #p2
        x6=self.elan2(x5)
        x7=self.pool2(x6) #p3
        x8=self.elan3(x7)
        x9=self.pool3(x8) #p4
        x10=self.elan4(x9)
        x11=self.neck(x10)
        x12=self.up1(x11) #p3
        x13=self.elan_1(x12,self.concatp3(x8))
        x14=self.up2(x13) #p2
        x15=self.elan_2(x14,self.concatp2(x6)) #out1_small
        x16=self.pool4(x15,x13)
        x17=self.elan_3(x16)   #out2_middle
        x18=self.pool5(x17,x11)
        x19=self.elan_4(x18)   #out3_huge
        out1=self.out_cov1(x15)#检测小尺度
        out2=self.out_cov2(x17)#检测中等尺度
        out3=self.out_cov3(x19)#检测大尺度
        y1=self.detect1(out1)
        y2=self.detect2(out2)
        y3=self.detect3(out3)

        return [y1,y2,y3]

#数据增强
train_transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def target_encode(strides,pred,tgt,anchors,nc):

    na=anchors.shape[1]#anchors[3,3,2]
    nt=len(tgt)#tgt[b,n,4]
    target_out=[]

    for i in range(nt):
        boxes=torch.stack(tgt[i]['boxes'],dim=0)
        labels=tgt[i]['labels']

        if len(boxes)==0:
            continue


        boxes_xyhw=torch.cat([
            (boxes[:,:2]+boxes[:,2:])/2,
            boxes[:,2:]-boxes[:,:2],
        ],dim=1)  #[n,4]绝对坐标,yolo

        for  si, (pre,anchor)  in  enumerate( zip( pred,anchors ) ):
            #print(pre.shape)
            _ , _ , H , W , _ = pre.shape
            stride=strides[si]
            scale=torch.tensor([stride,stride,stride,stride])
            boxes_scale=boxes_xyhw/scale  #归一化,[n,4]相对坐标,yolo

            boxes_wh=boxes_scale[:,2:] #boxes_wh[n,2]
            anchor_wh=anchor

            wh_ratio=boxes_wh[ : ,None, : ] / anchor [None, : , : ] #anchor[3,2],ratio[n,3,2]
            max_ratio=wh_ratio.max(dim=-1)[0]#[n,3]每一个标框和锚框匹配
            min_ratio=wh_ratio.min(dim=-1)[0]
            keep=(max_ratio<4.0)&(min_ratio>1/4.0)

            for j in range(boxes.shape[0]):
                if keep[j].sum==0 :
                    continue
                for ai in (keep[j].nonzero(as_tuple=False).flatten()):#把不同尺度的anchor拼到同一维
                    cx,cy,w,h=boxes_scale[j]

                    tx = cx - int(cx)
                    ty = cy - int(cy)
                    tw = torch.log(w / anchor_wh[ai,0])
                    th = torch.log(h / anchor_wh[ai,1])

                    tx = tx.clamp(1e-6, 1 - 1e-6)
                    ty = ty.clamp(1e-6, 1 - 1e-6)

                    target = torch.tensor([
                        i,  #batch图像索引
                        ai,  #anchor索引
                        int(tx), int(ty),  #中心坐标
                        tx, ty, tw, th,
                        labels[j].float()
                    ])
                    target_out.append(target)

    if len(target_out) == 0:
        return torch.zeros((0, 6))

    return torch.stack(target_out, dim=0)  #target_encoded[b*n,9]


def ciou(boxes1, boxes2):

    b1_x1, b1_y1, b1_x2, b1_y2 = boxes1.unbind(dim=-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2.unbind(dim=-1)

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

    inter = ((torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) *\
             (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0))

    union = w1*h1 + w2*h2 - inter + 1e-7
    iou = inter / union

    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2 = cw**2 + ch**2 + 1e-7

    b1_x=(b1_x1+b1_x2)/2; b1_y=(b1_y1+b1_y2)/2
    b2_x=(b2_x1+b2_x2)/2; b2_y=(b2_y1+b2_y2)/2
    center_distance = (b1_x - b2_x)**2 + (b1_y - b2_y)**2

    # Aspect ratio penalty
    v = (4 / (torch.pi**2)) * (torch.atan(w2/h2) - torch.atan(w1/h1))**2
    alpha = v / (1 - iou + v + 1e-7)

    return iou - (center_distance / c2 + alpha * v)


def YOLO_loss(preds, targets,anchors,nc=20,strides=[4,8,16],boxw=0.05, objw=1.0, clsw=0.5):

    device=preds[0].device
    lbox=torch.zeros(1,device=device)
    lobj=torch.zeros(1,device=device)
    lcls=torch.zeros(1,device=device)

    if(targets.shape[0]==0):

        for pred in preds:
            obj = pred[..., 4]
            lobj += (1 - obj.sigmoid()).pow(2).mean()
        lobj*=objw

        return lbox,lobj,lcls

    idx=targets[...,0].long()
    anchor_idx=targets[...,1].long()
    bx=targets[...,2].long(); by=targets[...,3].long()
    cx=targets[...,4]; cy=targets[...,5]
    tw=targets[...,6]; th=targets[...,7]
    labels=targets[...,8].long()

    for si,pred in enumerate(preds):
        stride=strides[si]
        anchor=anchors[si].to(device)
        b,na,h,w,c=pred.shape

        match = ((anchor_idx == si * 3) |
                 (anchor_idx == si * 3 + 1) |
                 (anchor_idx == si * 3 + 2))    #na=3

        if not match.any():
            obj=pred[...,4]
            lobj+=(1-obj.sigmoid()).pow(2).mean()
            continue

        bi=idx[match]
        gx=bx[match]; gy=by[match]
        tx=cx[match]; ty=cy[match]
        t_w=tw[match]; t_h=th[match]
        cls=labels[match]

        local_aidx = anchor_idx[match] - si * 3

        pred_xywh=pred[bi,local_aidx,gy,gx,:4]

        pred_x = (pred_xywh[:, 0].sigmoid() + gx) * stride
        pred_y = (pred_xywh[:, 1].sigmoid() + gy) * stride
        pred_w = pred_xywh[:, 2].exp() * anchor[local_aidx, 0] * stride
        pred_h = pred_xywh[:, 3].exp() * anchor[local_aidx, 1] * stride

        pred_boxes = torch.stack([
            pred_x - pred_w / 2,
            pred_y - pred_h / 2,
            pred_x + pred_w / 2,
            pred_y + pred_h / 2
        ], dim=-1)

        tgt_x=(tx+gx)*stride
        tgt_y=(ty+gy)*stride
        tgt_w=t_w.exp()*anchor[local_aidx, 0] * stride
        tgt_h=t_h.exp()*anchor[local_aidx, 1] * stride

        tgt_boxes = torch.stack([
            tgt_x - tgt_w / 2,
            tgt_y - tgt_h / 2,
            tgt_x + tgt_w / 2,
            tgt_y + tgt_h / 2
        ],dim=-1)

        lbox+=ciou(pred_boxes, tgt_boxes).mean()*boxw

        p_obj=pred[bi,local_aidx,gy,gx,4]
        lobj += F.binary_cross_entropy_with_logits(
            p_obj, torch.ones_like(p_obj), reduction='mean'
        ) * objw

        cls_pred = pred[bi, local_aidx, gy, gx, 5:]
        one_hot = torch.zeros_like(cls_pred)
        one_hot.scatter_(1, cls.unsqueeze(1), 1)
        lcls += F.binary_cross_entropy_with_logits(
            cls_pred, one_hot, reduction='mean'
        ) * clsw

    return lbox,lobj,lcls


#实例化及超参数设置
strides=[4,8,16];na=3;nc=20
anchors = [
    [[10,10],  [20,25],   [35,40]],   #小物体0
    [[12,12],  [22,28],   [40,50]],  #中物体1
    [[ 8,8 ],  [14,18],   [25,30]]  #大物体2
]
anchors = torch.tensor(anchors, dtype=torch.float32)
data_root="..\\data\\VOC2007"
n=len(os.listdir(os.path.join(data_root,"JPEGImages")))

train_ratio=0.7
val_ratio=0.15
epoch=100
lr=0.001;momentum=0.9;weight_decay=0.0001
batch_size=1

dataset=MyDataset(data_root,train=True,transform=train_transform)
ntrain=int(train_ratio*n);nval=int(val_ratio*n);ntest=n-ntrain-nval
train_data,val_data,test_data=random_split(dataset,[ntrain,nval,ntest])

def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return torch.stack(images), targets

train_loader=DataLoader(train_data,batch_size=batch_size,shuffle=True,collate_fn=collate_fn)
val_loader=DataLoader(val_data,batch_size=batch_size,shuffle=False,collate_fn=collate_fn)
test_loader=DataLoader(test_data,batch_size=batch_size,shuffle=False,collate_fn=collate_fn)

module=net(20,3)
optimizer = Adam(module.parameters(),lr=lr)


# def test():
#     loader=enumerate(train_loader)
#     for i,(x,target) in loader:
#         y=module(x)
#         for res in y:
#             print(res.shape)
#         break


def train(epoch):

    module.train()
    module.cuda()
    Times=[]
    Loss=[]

    for e in range(epoch):
        all_loss=0.0

        for i,(x,tgt) in enumerate(train_loader):
            x=x.cuda()
            optimizer.zero_grad()
            y = module(x)
            tgt = target_encode(strides,y, tgt, anchors, nc)
            tgt = tgt.cuda()
            lbox,lobj,lcls = YOLO_loss(y,tgt,anchors,strides)
            loss=lbox+lobj+lcls
            loss.backward()
            optimizer.step()
            all_loss += loss.item()

            if i % 50 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(e+1, epoch, i, len(train_loader), loss.item()))

        if e % 5 == 0:
            Loss.append(all_loss/len(train_loader))
            Times.append(e+1)

    torch.save(module.state_dict(),"module_save")

    x=np.array(Times)
    y=np.array(Loss)
    plt.plot(x,y,marker='o')
    plt.title('Loss during training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


train(epoch)