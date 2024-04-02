from commons_pt import *

class CYoloV9Backbone(nn.Module):
    def __init__(self, return_idx=[2,3,4]):
        super().__init__()
        self.return_idx = return_idx

        #real yolov9 params
        # self.pyramids = nn.Sequential(
        #     nn.Sequential(
        #         Silence(),
        #         Conv(3, 64, 3, 2),
        #     ),
        #     nn.Sequential(
        #         Conv(64, 128, 3, 2),
        #         RepNCSPELAN4(128, 256, 128, 64)
        #     ),
        #     nn.Sequential(
        #         Conv(256, 256, 3, 2),
        #         RepNCSPELAN4(256, 512, 256, 128, 1)
        #     ),            
        #     nn.Sequential(
        #         Conv(512, 512, 3, 2),
        #         RepNCSPELAN4(512, 512, 512, 256, 1)
        #     ),            
        #     nn.Sequential(
        #         Conv(512, 512, 3, 2),
        #         RepNCSPELAN4(512, 512, 512, 256, 1)
        #     ),                        
        # )        

        # number of params alike resnet50
        self.pyramids = nn.Sequential(
            nn.Sequential(
                Silence(),
                Conv(3, 64, 3, 2),
            ),
            nn.Sequential(
                Conv(64, 128, 3, 2),
                RepNCSPELAN4(128, 256, 128, 64)
            ),
            nn.Sequential(
                Conv(256, 256, 3, 2),
                RepNCSPELAN4(256, 512, 256, 128, 1)
            ),            
            nn.Sequential(
                Conv(512, 1024, 3, 2),
                RepNCSPELAN4(1024, 1024, 1024, 256, 1)
            ),            
            nn.Sequential(  
                Conv(1024, 1024, 3, 2),
                RepNCSPELAN4(1024, 1024, 1024, 256, 1)
            ),                        
        )            
            
        print('pyramids length:', len(self.pyramids))

    def forward(self, x):
        results = []
        for i in range(self.return_idx[-1]+1):
            pyr = self.pyramids[i]
            x = pyr(x)
            if i in self.return_idx:
                results.append(x)
        return results

def mulp(*x):
    res = x[0]
    for i in x[1:]:
        res*=i
    return res

if __name__=='__main__':
    device = 'cuda'
    x = torch.randn(1, 3, 640, 640)
    model = CYoloV9Backbone()
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)
    out = model(x)
    print([i.shape for i in out])
    print(([x for i in out for j in i for x in j.shape]))
    