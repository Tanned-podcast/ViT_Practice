#Cursorの使い方：Ctrl+Lでチャット，tabで補完，Ctrl+Kで生成 普通にTabで空白挿入したいときは補完をEscでキャンセルしてからTab
import torch
import torch.nn as nn
import torch.nn.functional as F

batch_size = 2
channels = 3
height = 32
width = 32
x = torch.randn(batch_size, channels, height, width)

#埋め込み後のベクトルの長さemb_dimはモデルの設計者が決めるんか？
#入力画像に対し下処理を行うInput Layer
#nn.Moduleを継承しておかないと，Pytorchのニューラルネットワークのパーツとして呼び出せない
#nn.Moduleのルールに従って，forwardメソッドを定義しておく
class VitInputLayer(nn.Module):
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
        #print(z0.shape)

        #先にConvくぐらしといて後からFlatten処理を行う　パッチの数が１次元になる
        #transpose(1, 2)でバッチサイズとパッチの数を入れ替える　バッチサイズ，パッチの数，埋め込み後のベクトル長さになる
        z0 = z0.flatten(2).transpose(1, 2)
        #print(z0.shape)

        #クラストークンのバッチサイズを揃え，z0と結合
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        z0 = torch.cat([cls_token, z0], dim=1)
        #print(z0.shape)

        #位置埋め込みを加算
        z0 = z0 + self.pos_embedding
        #print(z0.shape)

        return z0

input_layer = VitInputLayer()
z0 = input_layer.forward(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, emb_dim: int = 384, num_heads: int = 6, dropout: float=0):
        super(MultiHeadSelfAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads #D/h
        self.sqrt_dh = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)

        #クエリ，キー，バリュー行列に変換するための重み
        self.w_q = nn.Linear(emb_dim, emb_dim, bias=False)
        self.w_k = nn.Linear(emb_dim, emb_dim, bias=False)
        self.w_v = nn.Linear(emb_dim, emb_dim, bias=False)

        #最後分割したヘッドを結合しなおして，出力行列に変換するための重み
        self.w_o = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.Dropout(dropout))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        #zが入力行列 サイズ取得
        batch_size, num_patch, _ = z.shape

        #埋め込み
        q = self.w_q(z)
        k = self.w_k(z)
        v = self.w_v(z)

        #ヘッド分割　バッチサイズ，パッチの数，ヘッド数，ベクトルの長さの順
        q = q.view(batch_size, num_patch, self.num_heads, self.head_dim)
        k = k.view(batch_size, num_patch, self.num_heads, self.head_dim)
        v = v.view(batch_size, num_patch, self.num_heads, self.head_dim)

        #print(q.shape)

        #transposeでバッチサイズ，ヘッド数，パッチの数，ベクトルの長さの順にする
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        #転置してqkからAttention Weightを計算　バッチサイズ，ヘッド数，パッチの数，パッチの数の順になる
        k_t = k.transpose(2, 3)
        attn = F.softmax(torch.matmul(q, k_t) / self.sqrt_dh, dim=-1)
        print(attn.shape)
        
        #ドロップアウト
        attn = self.dropout(attn)

        #Attention Weightとバリュー行列の行列積 バッチサイズ，ヘッド数，パッチの数，ベクトルの長さの順になる
        out = torch.matmul(attn, v)
        #print(out.shape)

        #transposeでバッチサイズ，パッチの数，ヘッド数，ベクトルの長さの順にする
        #reshapeでヘッド分割してたのを結合　バッチサイズ，パッチの数，埋め込み後のベクトル長さの順にする
        out = out.transpose(1, 2)
        out=out.reshape(batch_size, -1, self.emb_dim)
        #print(out.shape)

        #最後の重みをかけて出力行列に変換
        out=self.w_o(out)
        #print(out.shape)

        return out


mhsa=MultiHeadSelfAttention()
out=mhsa.forward(z0)


#MHSAまで含めたEncoderBlockを作っていく
class ViTEncoderBlock(nn.Module):
    def __init__(self, emb_dim: int = 384, num_heads: int = 8, hidden_dim: int = 384*4, dropout: float=0):
        super(ViTEncoderBlock, self).__init__()

        #EncoderBlock内のレイヤー組み立て：LayerNorm，MultiHeadSelfAttention，LayerNorm，MLP
        self.ln1=nn.LayerNorm(emb_dim)
        self.mha=MultiHeadSelfAttention(emb_dim, num_heads, dropout)
        self.ln2=nn.LayerNorm(emb_dim)
        self.mlp=nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.Dropout(dropout)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        #zがInput Layerからの入力行列 これを作ったレイヤーに通す
        #途中でResidual Blockが２つあるので，２回に分けて処理する（かしこい）
        z=z+self.mha(self.ln1(z))
        z=z+self.mlp(self.ln2(z))

        return z

vit_encoder_block=ViTEncoderBlock()
z=vit_encoder_block.forward(z0)
print(z.shape)

