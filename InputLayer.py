#Cursorの使い方：Ctrl+Lでチャット，tabで補完，Ctrl+Kで生成 普通にTabで空白挿入したいときは補完をEscでキャンセルしてからTab
import torch
import torch.nn as nn

batch_size = 2
channels = 3
height = 32
width = 32
x = torch.randn(batch_size, channels, height, width)

#埋め込み後のベクトルの長さemb_dimはモデルの設計者が決めるんか？
class VitInputLayer():
    def __init__ (self, num_patch_row: int = 2, in_channels: int = 3, emb_dim: int = 384, img_size: int = 32):
        super(VitInputLayer, self).__init__()
        self.num_patch_row = num_patch_row
        self.in_channels = in_channels
        self.emb_dim = emb_dim
        self.img_size = img_size
        
        # パッチの数とパッチのサイズを計算
        self.num_patch = num_patch_row * num_patch_row
        self.patch_size = img_size // num_patch_row
        
        # 入力画像をパッチ分割し，パッチを埋め込み　畳み込みレイヤのフィルタをストライドさせることでパッチ分割してるし，
        # 畳み込み演算は実質埋め込んでるのと同じ
        self.patch_emb_layer = nn.Conv2d(in_channels, emb_dim, kernel_size=self.patch_size, stride=self.patch_size)

        # クラストークンと位置埋め込みの初期値を作成（標準正規分布からの乱数）
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patch + 1, emb_dim))

    #xは入力画像
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #バッチサイズ，チャンネル数，高さ，幅　=> バッチサイズ，埋め込み後のベクトル長さ，パッチの数（縦，横）になる
        z0 = self.patch_emb_layer(x)
        print(z0.shape)

        #先にConvくぐらしといて後からFlatten処理を行う　パッチの数が１次元になる
        #transpose(1, 2)でバッチサイズとパッチの数を入れ替える　バッチサイズ，パッチの数，埋め込み後のベクトル長さになる
        z0 = z0.flatten(2).transpose(1, 2)
        print(z0.shape)

        #クラストークンのバッチサイズを揃え，z0と結合
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        z0 = torch.cat([cls_token, z0], dim=1)
        print(z0.shape)

        #位置埋め込みを加算
        z0 = z0 + self.pos_embedding
        print(z0.shape)

        return z0

input_layer = VitInputLayer()
z0 = input_layer.forward(x)


