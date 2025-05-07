from ViT_Layers import VitInputLayer, ViTEncoderBlock
import torch.nn as nn
import torch

num_classes = 10
batch_size, channel, height, width = 2, 3, 32, 32

#テスト用の疑似画像を入力として作成
x = torch.randn(batch_size, channel, height, width)

#モデルの定義
class Vit(nn.Module):
    def __init__(self, in_channels: int = 3, img_size: int = 32, num_patch_row: int = 2, emb_dim: int = 384, hidden_dim: int = 384*4,
                 num_heads: int = 8, num_blocks: int = 7, num_classes: int = 10, dropout: float=0):
        super(Vit, self).__init__()

        #各レイヤ組み立て　encoder Blockは何重かに重ねる 最後CLSだけを抜き出して分類するので，CLSのベクトル長さemb_dimと出力するクラス数をmlp_headで設定しとく
        self.input_layer = VitInputLayer(num_patch_row, in_channels, emb_dim, img_size)
        self.encoder = nn.Sequential(*[ViTEncoderBlock(emb_dim, num_heads, hidden_dim = hidden_dim, dropout = dropout) for _ in range(num_blocks)])
        self.mlp_head = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, num_classes))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        #入力画像xをレイヤに通していく
        out = self.input_layer(x)
        print(out.shape) #(バッチサイズ，パッチの数+1，埋め込み後のベクトル長さ)の行列になってることを確認
        out = self.encoder(out)
        print(out.shape) #(バッチサイズ，パッチの数+1，埋め込み後のベクトル長さ)の行列になってることを確認

        #CLSだけを抜き出す　各クラスの確率を計算したpredが出てくる
        cls_token = out[:, 0]
        pred = self.mlp_head(cls_token)
        pred = torch.softmax(pred, dim=1)
        return pred

vit = Vit(in_channels = channel, num_classes = num_classes)
pred = vit(x)
print(pred.shape) #(バッチサイズ，クラス数)の行列になってることを確認
print(pred)
