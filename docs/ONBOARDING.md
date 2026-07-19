# 環境セットアップ・オンボーディングガイド

**作成日**: 2026-07-19 05:16 UTC

**対象**: 新しいセッション・エージェント、ZipMap の推論／評価を行う開発メンバー

**プロジェクト**: ZipMap (`yuki-inaho/ZipMap`)
**目的**: GPU 環境で ZipMap Streaming を再現し、Gradio を使わない連続 RGB 推論と pytest smoke test を実行できる状態にする。

---

## 目次

1. [プロジェクト概要](#1-プロジェクト概要)
2. [現在のプロジェクト状態](#2-現在のプロジェクト状態)
3. [前提条件の確認](#3-前提条件の確認)
4. [環境セットアップ手順](#4-環境セットアップ手順)
5. [動作確認](#5-動作確認)
6. [トラブルシューティング](#6-トラブルシューティング)
7. [次のステップ](#7-次のステップ)
8. [環境セットアップ完了チェックリスト](#8-環境セットアップ完了チェックリスト)
9. [更新履歴](#9-更新履歴)

---

## 1. プロジェクト概要

### プロジェクト名

**ZipMap**

複数視点 RGB 入力からカメラ姿勢・内部パラメータ・深度を推定するモデル群。Streaming 版は時系列 RGB を入力にした逐次推論用モデルである。

### 最終目標

順序付けた RGB 画像列に `checkpoint_online.pt` を適用し、Gradio を起動せずに姿勢・内部パラメータ・深度を出力する。あわせて、同梱サンプル動画と pytest により、モデルを実際にロードする GPU smoke test を再現可能にする。

### 主要コンポーネント

* `zipmap/models/ZipMap_AR.py` — Streaming ZipMap モデル本体
* `demo_gradio_zipmap_streaming.py` — 既存の対話デモ
* `scripts/run_zipmap_streaming_sequence.py` — Gradio 非依存の連続 RGB CLI
* `tests/test_streaming_smoke.py` — 同梱動画を用いる pytest smoke test
* `examples/videos/drift-straight.mp4` — smoke test の入力サンプル

### 実行フロー

~~~
連番付き RGB 画像列
  → run_zipmap_streaming_sequence.py
  → ZipMap Streaming (CUDA)
  → predictions.npz（W2C pose / K / depth / confidence）+ summary.json
~~~

## 2. 現在のプロジェクト状態

### 完了済み

| 分類 | 状態 | 説明 |
| --- | --- | --- |
| uv 環境 | 🟢 | Python 3.11 用の lockfile を生成し、`uv sync --group dev` で依存解決済み |
| CUDA 推論依存 | 🟢 | CUDA 12 系と整合する `onnxruntime-gpu<1.27` を指定 |
| Gradio 非依存 CLI | 🟢 | 連続 RGB を入力し、Streaming 推論結果を NPZ/JSON へ保存 |
| pytest | 🟢 | 通常テストと、明示実行する GPU smoke marker を定義 |
| GPU smoke | 🟢 | 同梱動画の連続2フレーム、Streaming checkpoint、CUDA で実行成功 |

### 依存パッケージのインストール状態

新しい環境では必ずリポジトリルートで `uv sync --group dev` を実行する。`checkpoints/` は Git 管理外であり、Streaming checkpoint は別途取得する必要がある。

この端末では共有環境 `/home/kasm-user/Desktop/venv/ZipMap` を使い、リポジトリの `.venv` はそこへのリンクになっている。ほかの端末ではこの配置を前提にせず、通常どおり `uv sync --group dev` でローカル `.venv` を作成してよい。

### 未実装・これから着手する項目

* 長い RGB 列を GPU メモリ量に合わせて chunk 化し、重複フレームで整合するバッチワークフロー
* 実データの姿勢・深度の定量評価と可視化
* checkpoint 不要の追加 CPU テスト、および GPU smoke test の自動実行方針

### 重要なファイル／ディレクトリ

~~~
/home/kasm-user/Desktop/ZipMap/
├── README.md
├── pyproject.toml                    # uv 依存関係と pytest 設定
├── uv.lock                           # 解決済み依存関係
├── checkpoints/                      # Git 管理外。モデル重みを置く
├── docs/
│   └── ONBOARDING.md                 # 本ドキュメント
├── examples/videos/drift-straight.mp4
├── scripts/run_zipmap_streaming_sequence.py
├── tests/test_streaming_smoke.py
└── zipmap/
    ├── models/ZipMap_AR.py
    └── utils/
~~~

## 3. 前提条件の確認

### 3.1 システム情報の確認

~~~bash
cat /etc/os-release | grep -E '^(NAME|VERSION)='
uname -r
nproc
free -h
pwd
nvidia-smi
~~~

期待値は、CUDA を利用可能な NVIDIA GPU と Python 3.11 以上である。推論は CUDA 必須であり、CPU のみの環境では GPU smoke test は実行できない。

### 3.2 必須ツールの存在確認

~~~bash
which uv && uv --version
which git && git --version
uv run python --version
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
~~~

`torch.cuda.is_available()` が `True` であることを確認する。checkpoint 取得前には `uv run hf --help` も確認する。

### 3.3 Git ブランチ・コミット確認

~~~bash
git branch --show-current
git log --oneline -1
git status --short
~~~

2026-07-19 時点の確認対象コミットは `e0f1f40 Correct license information`、ブランチは `main`。作業開始前に必ず `git status` を確認する。

## 4. 環境セットアップ手順

### 4.1 リポジトリと uv 環境

~~~bash
cd ~/Desktop/ZipMap
uv sync --group dev
~~~

`pytest` は dev dependency group に含まれる。lockfile と `pyproject.toml` に差異がある場合は、意図した依存更新かを確認してから `uv lock` を実行する。

### 4.2 Streaming checkpoint の取得

~~~bash
cd ~/Desktop/ZipMap
mkdir -p checkpoints
uv run hf download coast01/ZipMap checkpoint_online.pt --local-dir checkpoints
test -s checkpoints/checkpoint_online.pt
~~~

Hugging Face の認証が必要な場合だけ、利用者のシェル設定にあるトークンを読み込むか `hf auth login` を実行する。トークンの内容をログ、Markdown、コミットに出力してはならない。`checkpoints/` は `.gitignore` 済みである。

### 4.3 Gradio 非依存の連続 RGB 推論

入力ディレクトリ直下に `.png`、`.jpg`、`.jpeg`、`.bmp`、`.webp` を置く。ファイル名の辞書順で入力されるため、時系列は `000000.png` のようにゼロ埋め連番にする。

~~~bash
cd ~/Desktop/ZipMap
TORCH_COMPILE_DISABLE=1 uv run python scripts/run_zipmap_streaming_sequence.py \
  --input-dir /absolute/path/to/sequential_rgb \
  --checkpoint checkpoints/checkpoint_online.pt \
  --output-dir /absolute/path/to/zipmap_output \
  --window-size 1
~~~

生成物は次のとおり。

* `predictions.npz`
  * `frame_names`: 入力ファイル名
  * `extrinsics_world_to_camera`: world-to-camera 行列、形状 `(N, 3, 4)`
  * `intrinsics`: 内部パラメータ、形状 `(N, 3, 3)`
  * `depth`, `depth_conf`: 深度と信頼度
* `summary.json`: フレーム数、画像テンソル形状、実行設定、出力先

CLI は全入力画像を一度にロードする。長い列を渡す前は、まず `--max-frames 2` で動作確認し、VRAM に収まる長さへ区切る。複数chunkを単純連結した pose は別座標系になる可能性があるため、共通フレーム等による後段整合なしに絶対軌跡として使わない。

### 4.4 環境変数

通常の Streaming 推論と smoke test では、初回コンパイル待ちを避けるため次を付ける。

~~~bash
export TORCH_COMPILE_DISABLE=1
~~~

この変数を恒久的に `~/.bashrc` へ追加する必要はない。プロジェクトに固有のトークンや秘密情報も `.bashrc`、Git、本文書へ記録しない。

## 5. 動作確認

### 5.1 依存パッケージの確認

~~~bash
cd ~/Desktop/ZipMap
uv run python -c "import torch, onnxruntime; print(torch.__version__); print(torch.cuda.is_available()); print(onnxruntime.get_available_providers())"
uv run pytest --version
~~~

CUDA が利用できる場合、ONNX Runtime provider に `CUDAExecutionProvider` が表示される。

### 5.2 通常の pytest

~~~bash
cd ~/Desktop/ZipMap
uv run pytest -q
~~~

GPU marker はデフォルトで除外される。現状は同梱サンプル動画の存在・非空を確認するテストが通過し、GPU smoke test は deselect される。

### 5.3 実モデルを用いる GPU smoke test

~~~bash
cd ~/Desktop/ZipMap
TORCH_COMPILE_DISABLE=1 uv run pytest -m gpu -q
~~~

このテストは `examples/videos/drift-straight.mp4` から先頭2フレームを抽出し、`checkpoint_online.pt` で CLI を subprocess 実行する。終了後に pose、K、depth の形状と有限値を検証する。checkpoint がなければテストは skip される。別の重みを使う場合は `ZIPMAP_SMOKE_CHECKPOINT=/path/to/checkpoint.pt` を指定する。

### 5.4 CLI の最小動作確認

~~~bash
cd ~/Desktop/ZipMap
uv run python scripts/run_zipmap_streaming_sequence.py --help
~~~

`--input-dir`、`--checkpoint`、`--output-dir`、`--max-frames`、`--window-size` が表示されることを確認する。

## 6. トラブルシューティング

### 問題1: `uv sync` 後に ONNX Runtime を import できない

CUDA 13 向けの ONNX Runtime が解決されると、CUDA 12 環境で共有ライブラリ不足になることがある。本リポジトリでは `onnxruntime-gpu<1.27` として CUDA 12 に合わせている。

~~~bash
cd ~/Desktop/ZipMap
uv sync --group dev
uv run python -c "import onnxruntime; print(onnxruntime.__version__)"
~~~

### 問題2: `ZipMap Streaming inference requires CUDA`

GPU がコンテナへ公開されていない、またはドライバとの組合せに問題がある。

~~~bash
nvidia-smi
cd ~/Desktop/ZipMap
uv run python -c "import torch; print(torch.cuda.is_available())"
~~~

前者が失敗する場合はホスト／コンテナ設定を確認する。後者だけ失敗する場合は、解決した PyTorch の CUDA build を確認する。

### 問題3: smoke test が skip される

`checkpoint_online.pt` がない場合は仕様どおり skip される。取得後に再実行する。

~~~bash
cd ~/Desktop/ZipMap
uv run hf download coast01/ZipMap checkpoint_online.pt --local-dir checkpoints
TORCH_COMPILE_DISABLE=1 uv run pytest -m gpu -q
~~~

### 問題4: 長い画像列で CUDA out of memory

CLI は全フレームをまとめてモデルへ渡す。まず小さな範囲で試す。

~~~bash
TORCH_COMPILE_DISABLE=1 uv run python scripts/run_zipmap_streaming_sequence.py \
  --input-dir /absolute/path/to/sequential_rgb \
  --checkpoint checkpoints/checkpoint_online.pt \
  --output-dir /tmp/zipmap_smoke \
  --max-frames 2
~~~

必要なら上流で時系列をchunk化し、後段で座標系を整合するワークフローを実装する。

## 7. 次のステップ

### 7.1 実装前の確認事項

次の開発では、以下を設計書に明記する。

* フェーズ別タスク分解（入力整備、chunk推論、chunk間整合、出力検証）
* API／内部状態設計（chunkの共有フレーム、座標系、再開用manifest）
* テストシナリオ構成（CPU unit、GPU smoke、実データの小規模回帰）
* 完了条件（出力形式、有限値、座標系の定義、既知データでの品質基準）

### 7.2 最初の実務タスク

実データへ適用する前に、入力ファイルの順序とサイズを固定した manifest を作成する。

~~~bash
find /absolute/path/to/sequential_rgb -maxdepth 1 -type f | sort > /absolute/path/to/frames_manifest.txt
~~~

manifest の先頭・末尾と連番欠損を確認してから、短い `--max-frames` 実行を行う。

### 7.3 優先順位付きタスクリスト

1. `uv sync --group dev` と GPU smoke test の再現
2. 実データの時系列順・画像寸法・欠損を manifest 化
3. 小さい連続区間への CLI 適用と NPZ の有限値確認
4. VRAM 制約を測定し、chunk長と共有フレーム数を決定
5. chunk間の pose 座標系整合と深度／点群出力の設計
6. GPUを備える CI または定期実行環境での regression test 化

### 7.4 参考ドキュメント一覧

* `README.md` — 公式の checkpoint、Gradio demo、学習手順
* `scripts/run_zipmap_streaming_sequence.py` — 非対話 Streaming CLI
* `tests/test_streaming_smoke.py` — GPU smoke test の実装
* `pyproject.toml` — uv 依存関係と pytest marker 設定

### 7.5 便利なコマンド集

~~~bash
cd ~/Desktop/ZipMap
uv run pytest -q
TORCH_COMPILE_DISABLE=1 uv run pytest -m gpu -q
uv run python scripts/run_zipmap_streaming_sequence.py --help
git status --short
nvidia-smi
~~~

## 8. 環境セットアップ完了チェックリスト

* [ ] 想定した OS、CPU、メモリ、NVIDIA GPU を確認した
* [ ] `uv`、`git`、`nvidia-smi` が利用できる
* [ ] 正しい Git ブランチ・コミット・作業ツリー状態を確認した
* [ ] `uv sync --group dev` が完了した
* [ ] `torch.cuda.is_available()` が `True` である
* [ ] Streaming checkpoint を `checkpoints/checkpoint_online.pt` に配置した
* [ ] `uv run pytest -q` が成功した
* [ ] `TORCH_COMPILE_DISABLE=1 uv run pytest -m gpu -q` が成功した、またはcheckpoint不足による skip を確認した
* [ ] 実データは短い連続区間で CLI の出力を確認してから本実行する方針を理解した

## 9. 更新履歴

* 2026-07-19 05:16 UTC: 初版作成。uv 環境、CUDA 依存、Streaming CLI、pytest smoke test、実データ適用時の注意を記載。

---

## このドキュメントについて

本ガイドは、新しいセッションや新規参加者が短時間で同一の開発・実行環境を再現し、ZipMap Streaming の連続 RGB 推論を安全に引き継ぐためのものです。問題が起きた場合は、まず `git status`、CLI の標準出力、pytest の失敗ログ、`nvidia-smi` を確認し、モデル重みや秘密情報を issue・ログ・コミットへ含めないでください。
