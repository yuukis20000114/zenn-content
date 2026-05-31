---
title: "AIボイスカバーを「コード」で作る — Demucs + RVC + Optuna で音楽知識ゼロから挑んだ音声変換パイプライン"
emoji: "🎤"
type: "tech"
topics: ["rvc", "python", "optuna", "machinelearning", "audio"]
published: false
---

## はじめに

YouTubeでフレディ・マーキュリーのAIカバー動画を見るのにはまりました。コメント欄には「AIすごい」とついていて、具体的にどういうモデルで、何をすればこんな音源が作れるのかが気になって調べて、やってみました。

特に、自分は音楽的な知識が一切ないので、そんな状態からどこまでやれるか試してみました。コードはClaude Codeに相談しながら書いています。音響処理の知識がない部分は「こういうことがしたい」と伝えて、ライブラリの選定やパラメータの設計をやってもらいました。

題材は「チェスター・ベニントンが米津玄師のLemonを歌ったら」です。

:::message
▶ [最終カバー音源を再生（clip_chorus_after.wav）](https://github.com/yuukis20000114/zenn-content/raw/main/audio/clip_chorus_after.wav)
まずは完成した音源を聴いてみてください。この記事では、ここに至るまでの過程を解説します。
:::

## まずシンプルにやってみる

最初のアプローチは単純です。楽曲をボーカルと伴奏に分離し、ボーカルをChesterの声に変換して、伴奏と合わせ直します。

### ボーカル分離（Demucs）

Metaが開発したDemucsで楽曲を2トラックに分けます。

```python
def separate(input_path, output_dir, model_name="htdemucs_ft", device=None):
    model = get_model(model_name)
    model.to(device)

    wav, sr = torchaudio.load(str(input_path))
    if sr != model.samplerate:
        wav = torchaudio.functional.resample(wav, sr, model.samplerate)

    ref = wav.mean(0)
    wav = (wav - ref.mean()) / ref.std()
    with torch.no_grad():
        sources = apply_model(model, wav.unsqueeze(0).to(device),
                              split=True, overlap=0.25)
    sources = sources * ref.std() + ref.mean()

    # model.sources にステム名のリスト（例: ["drums", "bass", "other", "vocals"]）が格納されている
    source_names = model.sources
    vocals = sources[0, source_names.index("vocals")]
    accompaniment = sources.squeeze(0).sum(dim=0) - vocals
```

ボーカル以外のステムを全て足し合わせて1本の伴奏にしています。個別に取り出すとステム間の残差ノイズが累積するとClaude Codeに言われ、この方式にしました。

### 声質変換（RVC）

分離したボーカルをRVC v2でChesterの声に変換します。

```python
converter = RVCConverter(device="cuda:0", is_half=True)

result_path = converter.infer_audio(
    voice_model="chester_bennington",
    audio_path=str(input_path),
    f0_change=0,
    f0_method="rmvpe+",
    index_rate=0.85,       # 声質変換の強さ
    rms_mix_rate=0.20,
    protect=0.25,          # 子音の保護
    audio_format="wav",
)
```

`index_rate`を上げるほどChesterの声に近づきますが、日本語の子音が潰れやすくなります。`protect`を下げると変換が強まりますが、歌詞が聞き取りにくくなります。パラメータの意味はClaude Codeに教えてもらいつつ、聴き比べながら`index_rate=0.85`、`protect=0.25`に落ち着きました。「声の再現度」と「歌詞の聞き取りやすさ」のトレードオフです。

### 合わせてみた結果

変換したボーカルと伴奏をそのまま混ぜてみました。一応カバー音源にはなります。ただ、聴いてみると何かが足りません。声が平坦で奥に引っ込んでいて、臨場感がありません。YouTubeで聴いたフレディのAIカバーとは雲泥の差でした。

▶ [エンハンス前の音源を再生（clip_chorus_before.wav）](https://github.com/yuukis20000114/zenn-content/raw/main/audio/clip_chorus_before.wav)

## 何が足りないのか

音楽制作の知識がないのでピンと来ませんでしたが、Claude Codeに聞いてみると「レコーディングスタジオではボーカルにEQ、コンプレッション、リバーブなどの処理を重ねるのが普通で、RVCの出力にはそれが入っていない」とのことでした。

なるほど、仕上げが足りないわけです。それなら、その工程をコードで再現する後段を追加してみましょう。

## 7段FXチェーンで仕上げる

Claude Codeに「PythonでボーカルにFXチェーンをかけたい」と伝えたところ、Spotify社がOSSとして公開しているpedalboardを勧められました。VSTプラグインが不要で、Pythonから直接パラメータを渡せます。これを使って7段のエフェクトチェーンを組んでもらいました。

```
入力（モノラル）
  ↓ ① ノイズゲート
  ↓ ② EQシェイピング
  ↓ ③ コンプレッション
  ↓ ④ ディエッサー
  ↓ ⑤ サチュレーション
  ↓ ⑥ ダブリング（モノラル→ステレオ）
  ↓ ⑦ 空間系（リバーブ + ディレイ）
出力（ステレオ）
```

各ステージの構成と周波数帯域などの設定はClaude Codeが選定しました。それぞれの役割を簡単に説明します。

**① ノイズゲート** — RVC変換時に生じるアーティファクトを除去します。一定音量以下の信号をカットします。

**② EQシェイピング** — 4バンドで周波数特性を整えます。3.5kHzのプレゼンス帯域を+3.5dBブーストすると、声がミックスの中で前に出てきます。この帯域は人間の耳が敏感な領域です。

```python
plugins = [
    HighpassFilter(cutoff_frequency_hz=100.0),
    LowShelfFilter(cutoff_frequency_hz=250.0, gain_db=-2.0),
    PeakFilter(cutoff_frequency_hz=3500.0, gain_db=3.5, q=1.5),
    HighShelfFilter(cutoff_frequency_hz=10000.0, gain_db=1.0),
]
```

**③ コンプレッション** — 音量のばらつきを抑えます。ロックボーカルは音量差が大きいため、`ratio=4.5`と強めに圧縮しています。

**④ ディエッサー** — 歯擦音（サ行の「シュッ」という音）を抑えます。pedalboardに専用のディエッサーはないので、PeakFilter→Compressor→PeakFilterのサンドイッチで代替しました。

```python
def _apply_de_esser(audio, sr, p):
    freq = p["frequency_hz"]  # 6000Hz
    board = Pedalboard([
        PeakFilter(cutoff_frequency_hz=freq, gain_db=6.0, q=2.0),
        Compressor(threshold_db=p["threshold_db"],
                   ratio=p["ratio"], attack_ms=0.5, release_ms=30.0),
        PeakFilter(cutoff_frequency_hz=freq, gain_db=-6.0, q=2.0),
    ])
    return board(audio, sr)
```

+6dBで歯擦音帯域だけ持ち上げ、コンプレッサーの閾値を超えさせて圧縮し、-6dBで元に戻します。これは「マルチバンド・コンプレッサー」や「ダイナミックEQ」と呼ばれる手法に近いもので、特定帯域だけを選択的に圧縮する処理を汎用エフェクトの組み合わせで実現しています。

**⑤ サチュレーション** — 歪み成分を2割ほど混ぜて、声にロック特有のざらつきと太さを加えます。

**⑥ ダブリング** — RVCの出力はモノラルです。左右でピッチを±5セントずらし、ディレイも非対称（左15ms/右20ms）にして、ステレオに広げます。ピッチシフトで左右の波形が非相関化されるため、コムフィルタリングが起きにくくなっています。

```python
def _generate_doubles(audio_mono, sr, p):
    detune = p["detune_cents"] / 100.0
    left_board = Pedalboard([
        PitchShift(semitones=detune),
        Delay(delay_seconds=p["delay_l_ms"] / 1000.0, mix=1.0),
        Gain(gain_db=p["level_db"]),
    ])
    right_board = Pedalboard([
        PitchShift(semitones=-detune),
        Delay(delay_seconds=p["delay_r_ms"] / 1000.0, mix=1.0),
        Gain(gain_db=p["level_db"]),
    ])
    return np.stack([
        (audio_mono + left_board(audio_mono, sr)).squeeze(),
        (audio_mono + right_board(audio_mono, sr)).squeeze(),
    ], axis=0)
```

**⑦ 空間系** — リバーブとディレイで空間の広がりを付与します。

## パラメータをOptunaで自動最適化する

FXチェーンの構成はClaude Codeに任せましたが、各パラメータの最適値はどうすればいいでしょうか。Claude Codeにプリセット値を作ってもらうこともできますが、それが本当に最適かは分かりません。しかも「良い音かどうか」は主観的であり、そのまま目的関数にできません。

Claude Codeに相談したところ、「本物のChesterの歌声との音響特徴量の距離を目的関数にして、Optunaで探索する」というアプローチを提案されました。リファレンスとしてChesterの「Numb」「In The End」のボーカルを用意し、処理後の音声との距離を最小化する方向で自動探索させます。

### 距離の測り方

librosaで5種類の音響特徴量を抽出します。

| 特徴量 | 何を捉えるか | 距離の計算方法 |
|--------|-------------|---------------|
| MFCC | 音色そのもの。人間の聴覚特性に基づくメルスケールで算出する | フレームごとの平均ベクトル間のコサイン距離 |
| スペクトルコントラスト | 声の明瞭さ | フレームごとの平均ベクトル間のコサイン距離 |
| スペクトル重心 | 声の明るさ | 平均値同士の相対誤差（L1） |
| RMS | 音量感 | 平均値同士の相対誤差（L1） |
| スペクトルロールオフ | 高域成分の多さ | 平均値同士の相対誤差（L1） |

MFCCやスペクトルコントラストのように多次元ベクトルとなる特徴量にはコサイン距離を使い、スカラー値の特徴量には相対誤差を使っています。これらを重み付きで合成し、音色を直接反映するMFCCの重みを最大（45%）にしました。

```python
distance = (0.45 * mfcc_dist
          + 0.20 * contrast_dist
          + 0.15 * centroid_dist
          + 0.10 * rms_dist
          + 0.10 * rolloff_dist)
```

### Optunaで探索する

OptunaのTPEサンプラーで100トライアル回します。計算コスト削減のため、楽曲全体ではなく25%地点から15秒のクリップを使いました。

```python
def objective(trial):
    params = _create_trial_params(trial, base_params)
    processed = apply_chain(source_clip.copy(), TARGET_SR, params)
    proc_feat = compute_features(_to_mono_normalized(processed), TARGET_SR)
    distances = [feature_distance(proc_feat, ref_feat)
                 for ref_feat in ref_features_list]
    return float(np.mean(distances))

sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(objective, n_trials=100)
```

設定ファイルに`auto_optimize: true`と書くだけで、パイプライン実行時に自動でOptunaが走るようにしました。

```yaml
enhance:
  enabled: true
  preset: "chester_bennington"
  auto_optimize: true
  references:
    - "inputs/references/chester_bennington/numb/vocals.wav"
    - "inputs/references/chester_bennington/in_the_end/vocals.wav"
  optimize_trials: 100
```

▶ [Optuna最適化後の音源を再生（clip_chorus_after.wav）](https://github.com/yuukis20000114/zenn-content/raw/main/audio/clip_chorus_after.wav)

## ミックス

エンハンス済みのボーカルと伴奏を合成します。ボーカルの音量は伴奏のRMSから逆算して自動で合わせています。M/S処理でステレオ感を少し広げ、リミッターで音割れを防いで完成です。

```python
if vocal_gain_db is None:
    acc_rms = np.sqrt(np.mean(accompaniment**2))
    vocal_gain_db = compute_gain_db(vocals, target_rms=acc_rms * 1.2)

mixed = vocals * vocal_mix_ratio + accompaniment * (1 - vocal_mix_ratio)
```

## 所感

音楽の知識がなくても、Claude Codeに聞きながら「なぜそうするか」を1つずつ理解していく作業は楽しかったです。EQで3.5kHzを持ち上げると声が前に出る理由、コンプレッサーのattackを速くしすぎるとなぜポンピングが起きるのか。自分で聴いて違いを確かめながら進められるのが良かったです。

音楽知識がある人ならOptunaを使うまでもなく手で追い込めるかもしれません。自分にはそれができなかったからこそ、「リファレンスとの距離を最小化する」というアプローチが必要でした。Claude Codeに音響処理の設計を任せつつ、自分は耳で判断する。この役割分担がうまくはまりました。

品質面では限界もあります。RVCの根本的な制約として、日本語の子音追従性の問題があります。特に「さ行」「た行」の再現性が低く、歌詞を知らない状態で聞くと分からない箇所があります。また、MFCCベースの距離は音色の類似度を測れますが、「自然に聞こえるか」「感情が伝わるか」は評価できません。最終的な品質判断は人間の耳に頼るしかありません。

得た知見をまとめると以下のとおりです。

- RVCの出力品質は後処理で大きく変わります。分離→変換だけでは臨場感が出ません
- パラメータの手動調整が難しいなら、リファレンス音源との距離をOptunaで最小化するアプローチが使えます
- ただし、最終的に「良い音かどうか」を判断するのは人間の耳です
