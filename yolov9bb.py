from commons_pt import *

class YoloV9Backbone(nn.Module):
    def __init__(self, c=[3, 64, 128, 256, 512, 1024, 2048], return_idx=[1,2,3]):
        super().__init__()
        self.return_idx = return_idx

        # 3-> 64
        
        self.pyramids = nn.Sequential(
            nn.Sequential(
                Silence(),
                Conv(c[0], c[1], 3, 2)                
            )
        )        
        last_up_c_i = 1
        for ch_i in range(2, len(c), 2):
            curr_c = c[ch_i]
            prev_c = c[ch_i-1]
            if (curr_c==prev_c) or ((ch_i+1)==len(c)):
                self.pyramids.append(
                    nn.Sequential(
                        Conv(prev_c, curr_c, 3, 2),
                        RepNCSPELAN4(curr_c, next_c, curr_c, c[last_up_c_i], 1)
                    )
                )

                print(f'Conv{(prev_c, curr_c, 3, 2)}')
                print(f'RepNCSPELAN4{(curr_c, next_c, curr_c, c[last_up_c_i], 1)}')
            else:
                next_c = c[ch_i+1]
                self.pyramids.append(
                    nn.Sequential(
                        Conv(prev_c, curr_c, 3, 2),
                        RepNCSPELAN4(curr_c, next_c, curr_c, prev_c, 1)
                    )
                )
                last_up_c_i = ch_i
                print(f'Conv{(prev_c, curr_c, 3, 2)}')
                print(f'RepNCSPELAN4{(curr_c, next_c, curr_c, prev_c, 1)}')
        # self.yolov9bb_layers = nn.Sequential(self.pyramid0, self.pyramid1, self.pyramid2, self.pyramid3, self.pyramid4)
        print('pyramids length:', len(self.pyramids))
    def forward(self, x):
        results = []
        for i in range(self.return_idx[-1]+1):
            print('forward', i)
            pyr = self.pyramids[i]
            x = pyr(x)
            if i in self.return_idx:
                results.append(x)
        return results

if __name__=='__main__':
    x = torch.randn(1, 3, 640, 640)
    model = YoloV9Backbone()
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)
    print([i.shape for i in model(x)])