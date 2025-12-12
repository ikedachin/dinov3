# DINOv3 のモデルカード

DINOv3 は、多様な設定で特化型最先端モデルを上回る性能を示す汎用視覚基盤モデルのファミリーです。微調整なしで幅広いビジョンタスクで高品質の密な特徴量を生成し、以前の自己・弱教師あり基盤モデルを大きく上回る性能を発揮します。

## モデルの詳細

これらは、DINOv3 論文で説明された手法に従って学習された Vision Transformer（ViT）と ConvNeXt モデルです。12種類のモデルが提供されています：

- ウェブデータ（LVD-1689M）で事前学習された 10 モデル
  - 1つの ViT-7B をスクラッチで学習
  - ViT-7B から蒸留された ViT-S/S+/B/L/H+ 系モデル 5つ
  - ViT-7B から蒸留された ConvNeXt-{T/S/B/L} 系モデル 4つ
- 衛星データ（SAT-493M）で事前学習された 2 モデル
  - 1つの ViT-7B をスクラッチで学習
  - ViT-7B から蒸留された ViT-L 1つ

各 Transformer ベースのモデルは画像を入力として受け取り、クラス（CLS）トークン、パッチトークン（およびレジスタトークン）を返します。これらのモデルは ViT アーキテクチャに従い、パッチサイズは 16 です。入力画像が 224x224 の場合、1 個のクラストークン + 4 個のレジスタトークン + 196 個のパッチトークン = 合計 201 トークンになります（DINOv2 のレジスタ付きでは 1 + 4 + 256 = 261 トークンでした）。

モデルは、画像サイズがパッチサイズ（16）の倍数であればより大きな画像を受け付けます。もし倍数でない場合は、パッチサイズの最も近い小さい倍数にクロップされます。

### モデルの説明

- **開発者:** Meta AI
- **モデル種別:** Vision Transformer, ConvNeXt
- **ライセンス:** [DINOv3 ライセンス](https://ai.meta.com/resources/models-and-libraries/dinov3-license/)

### モデルのソース

- **リポジトリ:** https://github.com/facebookresearch/dinov3
- **論文:** https://arxiv.org/abs/2508.10104

## 利用方法

これらのモデルは、下流タスク向けの多目的バックボーンとして機能し、微調整なしで高い性能を発揮します。

### 直接利用

固定した特徴量をそのまま使い、簡単な下流分類器で競合する結果を得られます：

- クラストークンに対する k-NN 分類
- クラストークンに対するロジスティック回帰
- クラストークンとパッチトークンの平均に対する線形層
- 最近傍（Nearest Neighbors）を用いた画像検索
- 幾何学的・意味的な 3D キーポイント対応
- 深度推定、セマンティックセグメンテーション（線形層を使用）
- 教師なし物体発見
- ビデオのセグメンテーショントラッキング
- 小さな 4 層のアテンションプローブを用いたビデオ分類

### 下流での微調整

微調整は追加の性能向上をもたらす場合がありますが、まずは固定特徴量を利用することが推奨されます。微調整を行うと、特徴量のバイアスがラベルに適合して増幅される可能性があります。

## バイアス、リスク、制約

DINOv3 は DINOv2 や SEERv2 と比べて地理的公平性や多様性に関して比較的一貫した性能を示しますが、低所得地域バケットでの性能低下が観測されます（最高所得バケットとの差が見られます）。

DINOv3 は地域別に比較的良好なスコアを達成していますが、ヨーロッパとアフリカの間に相対的な差が残っています。

### 推奨

モデルを微調整すると特徴に偏りが増すため、必要でない限り固定した特徴を用いることが推奨されます。

## モデルの使い始め

以下のコードでモデルを使い始められます。

```python
import torch

model = torch.hub.load(
    repo_or_dir='facebookresearch/dinov3',
    model='<MODEL_NAME>',
    weights='<PATH/OR/URL/TO/CHECKPOINT>',
)

# MODEL_NAME の候補:
# - dinov3_vits16
# - dinov3_vits16plus
# - dinov3_vitb16
# - dinov3_vitl16
# - dinov3_vith16plus
# - dinov3_vit7b16
# - dinov3_convnext_tiny
# - dinov3_convnext_small
# - dinov3_convnext_base
# - dinov3_convnext_large

# 例:
dinov3_vits16 = torch.hub.load(
    repo_or_dir='facebookresearch/dinov3',
    model='dinov3_vits16',
    weights='<PATH/OR/URL/TO/DINOV3/VITS16/LVD1689M/CHECKPOINT>',
)
```

## トレーニングの詳細

### トレーニングデータ

- ウェブデータセット（LVD-1689M）：1,689M（約16.89億）画像を収集したキュレーション済みデータセット。元の大規模プールは 170 億のウェブ画像（公開投稿された Instagram 画像など）から抽出。

- 衛星データセット（SAT-493M）：Maxar の RGB 正射投影画像からサンプリングした 493M の 512x512 画像（解像度 0.6m）

### トレーニング手順

**学習目的：**

- DINO 自己蒸留損失（マルチクロップ）
- iBOT のマスク付き画像モデリング損失
- クラストークンに対する KoLeo 正則化
- Gram anchoring

- **トレーニング設定:** PyTorch FSDP2（bf16 と fp8 行列乗算を用いる）

**蒸留:**

蒸留は標準的な DINOv3 の事前学習手順に従いますが、教師モデルは固定された事前学習済み ViT-7B です。

## 評価

**結果**

評価プロトコルの詳細は論文を参照してください。

（以下に示すのは論文に載っている主要な評価結果の要旨です）

*ウェブ（LVD-1689M）で事前学習／蒸留した ViT バックボーンの結果の例*

（表は原文の数値を示します）

*ConvNeXt 系バックボーン（LVD-1689M で蒸留）の結果の例*

（表は原文の数値を示します）

*衛星データ（SAT-493M）で事前学習／蒸留した ViT バックボーンの結果の例*

（表は原文の数値を示します）

詳細な数値と実験条件は元の MODEL_CARD.md または論文を参照してください。

## 環境への影響

- **ハードウェア種別:** Nvidia H100
- **使用時間:** ViT-7B モデルの学習で 61,440 時間
- **クラウドプロバイダー:** プライベートインフラ
- **計算リージョン:** USA
- **排出炭素量:** 18 t CO2eq

## 技術仕様

### モデルアーキテクチャと目的関数

Vision Transformer モデル:

- ViT-S（21M パラメータ）: パッチサイズ 16、埋め込み次元 384、レジスタトークン 4、ヘッド数 6、MLP FFN、RoPE
- ViT-S+（29M）: パッチサイズ 16、埋め込み次元 384、レジスタトークン 4、ヘッド数 6、SwiGLU FFN、RoPE
- ViT-B（86M）: パッチサイズ 16、埋め込み次元 768、レジスタトークン 4、ヘッド数 12、MLP FFN、RoPE
- ViT-L（300M）: パッチサイズ 16、埋め込み次元 1024、レジスタトークン 4、ヘッド数 16、MLP FFN、RoPE
- ViT-H+（840M）: パッチサイズ 16、埋め込み次元 1280、レジスタトークン 4、ヘッド数 20、SwiGLU FFN、RoPE
- ViT-7B（6716M）: パッチサイズ 16、埋め込み次元 4096、レジスタトークン 4、ヘッド数 32、SwiGLU FFN、RoPE

ConvNeXt モデル:

- ConvNeXt Tiny（29M）
- ConvNeXt Small（50M）
- ConvNeXt Base（89M）
- ConvNeXt Large（198M）

### 計算インフラ

#### ハードウェア

Nvidia H100 GPU

#### ソフトウェア

PyTorch 2.7

## 参考情報

[ブログ投稿](https://ai.meta.com/blog/dinov3-self-supervised-vision-model/) と公式サイト（https://ai.meta.com/dinov3/）を参照してください。

## 引用

**BibTeX**

```
@misc{simeoni2025dinov3,
  title={{DINOv3}},
  author={Sim{\\'e}oni, Oriane and Vo, Huy V. and Seitzer, Maximilian and Baldassarre, Federico and Oquab, Maxime and Jose, Cijo and Khalidov, Vasil and Szafraniec, Marc and Yi, Seungeun and Ramamonjisoa, Micha{\\"e}l and Massa, Francisco and Haziza, Daniel and Wehrstedt, Luca and Wang, Jianyuan and Darcet, Timoth{\\'e}e and Moutakanni, Th{\\'e}o and Sentana, Leonel and Roberts, Claire and Vedaldi, Andrea and Tolan, Jamie and Brandt, John and Couprie, Camille and Mairal, Julien and J{\\'e}gou, Herv{\\'e} and Labatut, Patrick and Bojanowski, Piotr},
  year={2025},
  eprint={2508.10104},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2508.10104},
}
```
