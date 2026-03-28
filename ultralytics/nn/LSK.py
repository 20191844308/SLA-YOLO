import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv
from ultralytics.nn.others.strippooling import *

from ultralytics.nn.modules import Conv
from ultralytics.nn.modules import Bottleneck

class LSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim//2, 1)
        self.conv2 = nn.Conv2d(dim, dim//2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim//2, dim, 1)
        self.mpm = MPMBlock(dim, [6,6])

    def forward(self, x):   
        attn1 = self.conv0(x)
        #print(attn1.size())
        attn2 = self.conv_spatial(attn1)
        #print(attn1.size())
        attn1 = self.conv1(attn1)
        #print(attn1.size())
        attn2 = self.conv2(attn2)
        
        attn = torch.cat([attn1, attn2], dim=1)
        #attn = self.mpm(attn)
        #print(attn.size())
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        #print(avg_attn.size())
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        #print(max_attn.size())
        agg = torch.cat([avg_attn, max_attn], dim=1)
        #print(agg.size()) 
        sig = self.conv_squeeze(agg).sigmoid()
        #print(sig.size())
        attn = attn1 * sig[:,0,:,:].unsqueeze(1) + attn2 * sig[:,1,:,:].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn

class C2f_LSK(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""
    # C2f与C2相比，每个Bottleneck的输出都会被Concat到一起。
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        # 假设输入的x大小是(b,c1,w,h)
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.lsk = LSKblock(c2)
        ## n个Bottleneck组成的ModuleList,可以把m看做是一个可迭代对象
    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1)) #torch.chunk(块数，维度)
        # cv1的大小是(b,c2,w,h)，对cv1在维度1等分成两份（假设分别是a和b），a和b的大小均是(b,c2/2,w,h)。此时y=[a,b]。
        y.extend(m(y[-1]) for m in self.m)
        # 然后对列表y中的最后一个张量b输入到ModuleList中的第1个bottleneck里，得到c,c的大小是(b,c2/2,w,h)。然后把c也加入y中。此时y=[a,b,c]
        # 重复上述操作n次（因为是n个bottleneck），最终得到的y列表中一共有n+2个元素。
        return self.lsk(self.cv2(torch.cat(y, 1)))
        # 对列表y中的张量在维度1进行连接，得到的张量大小是(b,(n+2)*c2/2,w,h)。
        # 最终通过cv2,输出张量的大小是(b,c2,w,h)

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1)) # 将y的第一维度拼接在一起（因为上面chnk是在第一维度分割）
    
if __name__ == "__main__":
    model = LSKblock(320)
    #model = C2f_LSK(320,512)
    inputs = torch.ones([2, 320, 256, 256]) #[b,c,h,w]
    outputs = model(inputs)
    print(outputs.size())