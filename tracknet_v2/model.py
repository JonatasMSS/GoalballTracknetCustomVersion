

import torch
import torch.nn as nn



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class TrackentV2(nn.Module):
    def __init__(self, input_channels=9, mode="train"):
        super(TrackentV2, self).__init__()
        self.input_channels = input_channels
        self.set_mode(mode)
        #Bloco 1
        self.conv1 = ConvBlock(input_channels, 64)
        self.conv2 = ConvBlock(64, 64)
        

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 


        ## Bloco 2
        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 128)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)


        # Bloco 3
        self.conv5 = ConvBlock(128, 256)
        self.conv6 = ConvBlock(256, 256)
        self.conv7 = ConvBlock(256, 256)

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        
        self.bottleneck = nn.Sequential(
            ConvBlock(256, 512), #8
            ConvBlock(512, 512), #9
            ConvBlock(512, 512) # 10
        )

        #Nearest -> Ele funciona pegando o pixel mais próximo e replicando ele para preencher a nova dimensão.
        #Bilinear -> Ele faz uma interpolação linear entre os pixels vizinhos para calcular o valor do pixel na nova dimensão. Mas ele pode introduzir artefatos de suavização, especialmente em imagens com bordas nítidas.
        self.up1 = nn.Upsample(
            scale_factor=2, mode='nearest'  
        )

        #Decoder 1
        self.conv11 = ConvBlock(512 + 256, 256) #10 
        self.conv12 = ConvBlock(256, 256) #12
        self.conv13 = ConvBlock(256, 256) #13

        self.up2 = nn.Upsample(
            scale_factor=2, mode='nearest'
        )

        # Decoder 2

        # 256 + 128 = 384
        self.conv14 = ConvBlock(384, 128)
        self.conv15 = ConvBlock(128, 128)


        #Decoder 3 

        self.up3 = nn.Upsample(
            scale_factor=2,
            mode="nearest"
        )

        # 128 + 64 = 192

        self.conv16 = ConvBlock(192, 64)
        self.conv17 = ConvBlock(64, 64)

        self.out = nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0)

        self.sig = nn.Sigmoid()

    def set_mode(self, mode):
        if mode not in {"train", "inference"}:
            raise ValueError("mode must be 'train' or 'inference'")
        self.mode = mode

    def forward(self,x):

        #Bloco 01
        en1 = self.conv1(x)
        en2 = self.conv2(en1)
        pool1 = self.pool(en2)

        #Bloco 02
        en3 = self.conv3(pool1)
        en4 = self.conv4(en3)
        pool2 = self.pool2(en4)

        # Bloco3

        en5 = self.conv5(pool2)
        en6 = self.conv6(en5)
        en7 = self.conv7(en6)
        pool3 = self.pool3(en7)

        #Bottleneck
        bottleneck = self.bottleneck(pool3)

        up1 = self.up1(bottleneck)
        concat1 = torch.cat((up1, en7), dim=1)
        dec1 = self.conv11(concat1)
        dec2 = self.conv12(dec1)
        dec3 = self.conv13(dec2)

        up2 = self.up2(dec3)
        concat2 = torch.cat((up2, en4), dim=1)

        dec4 = self.conv14(concat2)
        dec5 = self.conv15(dec4)

        up3 = self.up3(dec5)
        concat3 = torch.cat((up3, en2), dim=1)
        dec6 = self.conv16(concat3)
        dec7 = self.conv17(dec6)

        output = self.out(dec7)
        if self.mode == "inference":
            return self.sig(output)
        return output


    def show_model_summary(self, input_size=(9, 128, 128)):
        from torchsummary import summary

        summary(self, input_size=input_size)


    

