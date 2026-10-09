# Adversarial panel — factual-accuracy review (2026-09-14)

Panel: Codex CLI 0.145.0 (GPT family) + claude-opus (Agent). 2 panelists, Round 1 blind, Round 2+3 merged. Both panelists reproduced claims by opening primary PDFs, running pytest, and re-executing benchmark functions.

## Codex Round 1
## A. カテゴリ別確度

採点は、**未検証項目を減点せず、実際に確認できた一致・不一致に基づく評価**とする。監査表のラベルは、修正後原稿ではなく、監査表に記録された元の主張に対して再判定した。

ファイル名は `/Users/deeptell01/Documents/alterego/personal/cqec/` 基準。一次資料の頁番号は、原則としてローカルPDFの印字頁を示す。

| カテゴリ | 確度 |
|---|---:|
| (1) 理論系引用の位置・番号 | **95/100** |
| (2) 応用系引用・書誌情報 | **85/100** |
| (3) 数学的主張 | **90/100** |
| (4) 数値・コード主張 | **70/100** |
| (5) 監査ラベルの妥当性 | **90/100** |

### (1) 理論系引用の位置・番号：95/100

一次PDFを開き、次を確認した。

| 出典 | 再確認した位置・内容 | 判定 |
|---|---|---|
| Watrous, *The Theory of Quantum Information* | §1.1.3、(1.143)–(1.144)：pp.26–27、Jordan–Hahn分解。(1.173)：p.33、Schatten双対性。(1.182)：p.34、ユニタリによる変分表示 | 修正後原稿は一致 |
| Watrous Theorem 1.14 | p.50、**Jensenの不等式** | 監査による旧引用の訂正は正しい |
| Fuchs–van de Graaf | Theorem 1・(46)：p.17。\(B\) の定義は Proposition 4・(24)：p.13 | \(B=\sqrt F\) とした修正は正しい |
| Życzkowski–Sommers | §II.C、(2.8)：p.3、(2.9)：p.4。Appendix Aはpp.11–12、複素Gaussian正規化と一様単体分布の説明はp.12 | 分布の引用は正しい。三つのモーメントそのものの引用ではない |
| Shiraishi–Takagi | Theorem 2：本文p.3。Theorem 3：本文p.4。Def. S.5/S.6：(4)/(5)、補遺p.4。Theorem S.24：補遺p.15 | 主要番号は一致 |
| Shiraishi Def. S.2 | **補遺p.3**、PDF通し10頁。時間並進共変性の等式を直接述べるのは、その直後の **Proposition S.3・(3)** | C08の「補遺p.2」は誤り |
| Marvian–Spekkens | Proposition 1・(2.11)：p.6、\(\mathrm{Modes}(\sigma)\subseteq\mathrm{Modes}(\rho)\) | 引用位置は一致 |
| Haahほか | Table I：p.2、Theorem 3：p.7、Theorem 4：pp.7–8 | 位置は概ね一致。ただし独立測定下限の対数因子が監査表で脱落 |

特に、Haahの独立測定に対するinfidelity下限は、Theorem 4のp.8では
\[
n\ge C\frac{dr^2}{\delta^2\ln(2/\delta)}(1-\delta)^4
\]
であり、C27等の注記にある、対数因子を付けない \(\Omega(dr^2/\delta^2)\) は厳密な転記ではない。

また、ShiraishiのTheorem 3は **有理係数で生成した \(C'\)** に関する必要条件であり、整数係数の \(C\) による完全な必要十分条件ではない。この点は監査表C03が正しく指摘している。

**反証条件：** 同じローカルPDFで、上記の一致判定箇所に別の定理・式が存在する、またはDef. S.2が実際には補遺p.2に掲載されていることが確認されれば、私の位置照合と95点の採点は誤りである。

### (2) 応用系引用・書誌情報：85/100

**確認できた内容**

- **rs-9366084**：実タイトルは *Biologically Plausible Quantum Error Correction in a Three-Layer Neural Spin Model*、著者は **Hikaru Wakaura単著**。修正後bibのタイトル・著者は一致する。ただしC45の「12 Apr 2026」は、取得済みPDFの掲載日表示と不一致。表紙は **April 13th, 2026**、本文の原稿日付は **April 9, 2026**。
- **Clinton**：ノートの取得元は `arXiv:2410.21517`。PDFはv2で、著者は **Laura Clinton、Toby S. Cubitt、Raul Garcia-Patron、Ashley Montanaro、Stasja Stanisic、Maarten Stroeks**。修正後bibの著者列は一致する。
- **PQES**：取得済みPDFは **2603.11568v2、2026年7月30日**。タイトルは *Hybrid Quantum Error Correction and Mitigation by Purification*。全文記録を利用し、postselectionをしないという説明はabstractと一致する。閾値 \(3/4\)、\(1/2\) も確認した。ただし前者は**検討された積状態族への局所depolarizing noise**、後者は **\(|+\rangle^{\otimes M}\) への局所dephasing**という条件付きである。
- **Unified**：全文で大文字の `ICEC` は0件。小文字の `icec.py` は2件ある。§6.13・Table 14、p.14の触媒周辺忠実度には **0.480、0.569、0.756、0.764、0.844**があり、要約「0.48–0.84」は妥当。触媒を保存することには成功していないというLimitationsの修正も一致する。
- **Unified §7.4、p.17**：Regevの標的状態を知ることが因数分解の知識を含む、という循環性の指摘を確認した。
- **Emerson 2005**：PDF全文をUnicode正規化・大文字小文字非区別で検索しても `randomized benchmarking` は0件。一方、abstractには平均ゲート忠実度の直接推定が記載されている。「平均ノイズ強度の推定」と「dephasing率の数％精度での同定」を同一視しない監査方針は正しい。

**残る不一致**

1. **QKANの具体例はUnifiedと同一ではない。**  
   Unified §5.1、p.8は \(x=0.5\) における
   \[
   (T_0,T_1,T_2,T_3)=(1,0.5,-0.5,-1)
   \]
   と次数2への打切りを説明する。現原稿・実コードは `target_func='sin'`、\(\sin(\pi x)\) の例である。原稿398行のsinという記述はコードに合うが、394行の「以下の具体例はUnifiedのもの」という帰属が合わない。
2. **PQESをv2で監査しているのに、bibの `eprint` とURLはv1のまま。** タイトルはv2と一致するが、監査した版への参照になっていない。
3. **Haah文献の著者名に誤字が残る。** `Harrow, Arllen W.` ではなく、PDF冒頭は **Aram W. Harrow**。

なお、QKANの取得済みv2の題名には *with Applications in Machine Learning and Multivariate State Preparation* が続く。bibの短い題名・2024年表記を初版への引用として扱うなら、**v1を明示してその版を照合する必要がある**。ここでは、未取得のv1の書誌を誤りとは断定しない。

**反証条件：** Unified §5.1に実際に今回と同じsin近似例が存在する、Haahの署名がArllenである、またはPQESの参照先が既にv2に統一されていることが同じファイルから確認されれば、これらに基づく減点は誤りである。

### (3) 数学的主張：90/100

**Lemma 1、Theorem 2、Corollary 1の証明自体は、記載された写像と仮定の下で正しい。** 減点対象は、それらを物理的到達可能性や全ベンチマークへ拡張した説明である。

#### Lemma 1の各ステップ

`paper_blind_cqec_qst.tex:190` の証明を確認した。

1. \(A=A_+-A_-\)、\(s=\operatorname{tr}A_-\) より、\(\operatorname{tr}A_+=1+s\)。正しい。
2. \(\Pi(A)-A=A_--sA_+/(1+s)\)。両項の支持が直交するので、ノルムは **\(2s\)**。正しい。
3. \(Y=I-2P_-\) は許容される変分変数であり、
   \[
   \operatorname{tr}Y(A-\sigma)
   =2s+2\operatorname{tr}(P_-\sigma)\ge2s.
   \]
   よって \(\|A-\sigma\|_1\ge2s\)。正しい。
4. 三角不等式から
   \[
   \|\Pi(A)-\sigma\|_1
   \le2s+\|A-\sigma\|_1
   \le2\|A-\sigma\|_1.
   \]
   正しい。

独立に生成した5,000組でも、ステップ2の等式残差は最大 \(7.1\times10^{-15}\)、ステップ3・4の違反はなかった。数値検査だけで証明とみなしたのではなく、上記の代数も確認した。

#### Theorem 2とCorollary 1

Theorem 2の導出は
\[
1-F
\le\|\Pi(A)-\rho_t\|_1
\le2\|A-\rho_t\|_1
\le2\Delta_{\rm mode}+2\|\rho_e-\rho_t\|_1
\]
で成立する。\(\Delta_{\rm mode}=0\) なら \(\Pi(A)=\rho_e\) なので、係数1への改善も正しい。

6,000例の数値検査で違反はなく、\(\Delta_{\rm mode}=0\) の1,614例では、出力と推定状態のFrobenius距離は最大 \(1.1\times10^{-15}\) だった。

Corollary 1の独立性による
\[
\mathbb E\|\bar\rho_n-\bar\rho\|_2^2
=\frac1n\mathbb E\|\rho^{(1)}-\bar\rho\|_2^2
\]
と、その後のCauchy–Schwarz・Jensenも正しい。\(O(n^{-1/2})\) は**固定次元での主張**である。\(d=4\) のHaar純粋状態を平均した独立検査でも、\(n=1,4,16,64\) に対する \(n\mathbb E\|\bar\rho_n-I/4\|_2^2\) はそれぞれ **0.750、0.752、0.747、0.759**で、理論値0.75と整合した。

#### 二側clipping不等式の3×3反例

自身で再計算した、簡単な反例は
\[
A=\begin{pmatrix}
0.6&0&0.3\\
0&0.3&0\\
0.3&0&0.1
\end{pmatrix},\qquad
B=\begin{pmatrix}
0.65&0&0.3\\
0&0.25&0\\
0.3&0&0.1
\end{pmatrix}.
\]
いずれもHermitian、trace 1であり、
\[
\|A-B\|_1=0.1000000000,\qquad
\|A_+-B_+\|_1=0.1048549987.
\]
したがって、主張された**二側非拡大性は偽**である。「偽」という修正は正しい。Davies全文にこの具体例が載っていると推測したわけではない。

#### \(\mathcal M\) がPSDを保存しない反例

\[
\rho_n=
\begin{pmatrix}
1/3&0.1&0\\
0.1&1/3&0.1\\
0&0.1&1/3
\end{pmatrix},\qquad
\rho_e=\frac13
\begin{pmatrix}1&1&1\\1&1&1\\1&1&1\end{pmatrix}.
\]
\(\rho_n\) の固有値は **0.191912、0.333333、0.474755**で正である。一方、
\[
\mathcal M_{\rho_n}(\rho_e)=\frac13
\begin{pmatrix}1&1&0\\1&1&1\\0&1&1\end{pmatrix}
\]
の最小固有値は
\[
\frac{1-\sqrt2}{3}=-0.1380711875.
\]
よってPSD非保存は確認できた。

さらに重要なのは、**その後のPSD射影が、消した成分を再生成する**ことだ。この例では
\[
[\Pi(\mathcal M(\rho_e))]_{02}=0.03033008589.
\]
したがって「最終出力でも削除した成分が必ずゼロ」という解釈は成立しない。

#### Haar/Dirichletモーメント

複素Gaussianベクトルを正規化してHaar状態を作り、各次元30万標本で検査した。

| \(d\) | 実測 \(\mathbb E[p_1]\) | 実測 \(\operatorname{Var}(p_1)\) | 実測 \(\mathbb E[p_1p_2]\) |
|---:|---:|---:|---:|
| 2 | 0.500166 | 0.0832423 | 0.166758 |
| 3 | 0.333547 | 0.0555042 | 0.0833894 |
| 4 | 0.249743 | 0.0374879 | 0.0499056 |
| 16 | 0.0625959 | 0.00346658 | 0.00368190 |
| 64 | 0.0156545 | 0.000236433 | 0.000241667 |

三つの理論値
\[
\frac1d,\qquad \frac{d-1}{d^2(d+1)},\qquad \frac1{d(d+1)}
\]
からの偏差は、全15項目で最大 **1.75標準誤差**だった。値は正しい。

**反証条件：** 上記の固定3×3行列でclipping差が0.1以下になる、\(\mathcal M(\rho_e)\) の最小固有値が非負になる、またはLemma/Theoremの仮定を満たす状態で不等式違反が再現されれば、私の数学検証と採点は誤りである。

### (4) 数値・コード主張：70/100

**数値の転記・再現性は良好だが、「何を計算した数値か」という説明に重大な不一致がある。**

#### 確認できたチェック項目

- **91 tests pass：再現した。**  
  通常の `python3` はpytest未導入、Anacondaの通常実行は一時ファイル作成で失敗した。次の書き込み不要設定では **91 passed in 13.35s**：
  ```text
  PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/bin/python -m pytest -q -o addopts= --capture=sys -p no:cacheprovider
  ```
- **`results/*.json`：8件。** 全件JSONとして読み込めた。
- **REPRODUCE対応：** 現原稿の17個の図ファイル名、8個の表ラベルはすべて記載され、対応スクリプトも存在する。
- **Table 2：4値ともJSONと一致する。**

| Algorithm | 原稿ChInv | JSONのChInv回復忠実度 |
|---|---:|---:|
| qDRIFT | 0.955 | 0.955054149 |
| QKAN | 0.986 | 0.985856502 |
| Control-free QPE | 0.956 | 0.956427986 |
| Regev | 0.738 | 0.738349784 |

描画部分だけを無効化し、実際の数値計算関数を再実行したところ、アルゴリズムベンチマークの**84条件・336数値は保存JSONと差0**だった。

#### 重大な不一致①：Table 2はTheorem 2の写像だけでは生成されない

`benchmark_blind_algorithms.py:193` はまず \(\Pi\circ\mathcal M\) を適用するが、続いて197–199行で **`ICECProtocol.correct` を追加実行する**。その内部には `icec.py:303` の有限コピー補間がある。

同じアルゴリズム状態・同じノイズで、推定値を真の標的に固定した再計算結果は：

| Algorithm | 原稿の式 \(\Pi(\mathcal M(\rho_t))\) | 実際の表生成経路 |
|---|---:|---:|
| qDRIFT | **1.000000** | 0.955054 |
| QKAN | **1.000000** | 0.985856 |
| Control-free QPE | **1.000000** | 0.956428 |
| Regev | **0.772481** | 0.738350 |

最初の3例では、削除された標的成分は0、すなわち \(\Delta_{\rm mode}=0\) だった。

**これはTheorem 2自体への反例ではない。全シミュレーションが同じ式を実装する、という原稿の説明への反例である。**

#### 重大な不一致②：Sec. 5.10のQEM比較はモード制限を省いている

`benchmark_blind_qem_compare.py:115` は：
```python
def icec_recover(noisy, target_est):
    return psd_project(target_est)
```
であり、`noisy`を使用せず、\(\mathcal M\) を実行しない。

同スクリプトの関数定義をそのまま読み込み、描画・保存を実行せず、同じseed・各次元15状態で再計算した。

| 次元 | QEMスクリプトのBlind ChInv | 同じ推定値にパッケージの \(\Pi\circ\mathcal M\) を適用 |
|---:|---:|---:|
| 32 | 約1.000000 | **0.592606** |
| 64 | 約1.000000 | **0.293345** |

したがって「全次元で1.000」は保存データとも計算とも一致するが、**原稿で定義したモード制限付き回復を使った結果ではない**。

Sec. 5.10–5.11が、比較はresource-matchedではないこと、密度行列へのoracleアクセスを仮定すること、測定コストを除外していることを明記した点は適切。しかし、その限定だけでは、回復写像の違いは解消しない。

#### 表記・PDF

- `ICEC`：**対象tex・bib・PDF本文では0件**。パッケージのPythonコードには3件あり、プロジェクト全体で0件ではない。
- `TODO / FIXME / TBD / XXX / ?? / PLACEHOLDER`：対象tex・bibで0件。
- **PDFは36頁。** 既存ログにも36頁とある。
- ただし既存ログには、0.84508ptのoverfull hbox、float-only page、hyperrefのPDF文字列警告がある。「compiles clean」を「警告も0」と読むなら一致しない。

**反証条件：** 同じファイル・入力で、追加補間なしの \(\Pi\circ\mathcal M\) が最初の3例に0.955/0.986/0.956を返す、またはQEMスクリプトが実際にはモード制限を通していることが確認されれば、私の主要な減点理由は誤りである。

### (5) 監査ラベルの妥当性：90/100

選択は、結果を見て選び直さず、
```python
sorted(random.Random(20260914).sample(range(1, 48), 12))
```
で行った。抽出結果は **C01、C13、C19、C23、C27、C30、C34、C35、C37、C40、C43、C44**。

| ID | 監査ラベル | 再判定・一次資料 |
|---|---|---|
| C01 | NF | **NF／検証不能**。Nielsen–Chuang全文なし |
| C13 | SS | **SS**。Unified §2.3、p.3、(7)の標的依存回復 |
| C19 | SS | **SS**。FvdG Theorem 1・(46)、p.17 |
| C23 | SS | **SS**。ZS §II.C、pp.3–4、Appendix A、p.12 |
| C27 | SS〈対数因子を除く〉 | **その限定付きでSS**。Haah Table I、p.2、Theorem 3、p.7 |
| C30 | PS | **PS**。Ivashkov p.5のChebyshev構成は支持。Fig.10、p.19はGaussian状態準備で、今回のsin具体例を支持しない |
| C34 | SS | **IOについてSS**。Baumgratz (2)/(9)、pp.1/3。Streltsov (38)、p.11。DIO/MIO反例の注意はp.12 |
| C35 | PS | **PS**。Haah p.2の \(1/n\) は先行研究の紹介 |
| C37 | SS | **SS**。Temme/Li/EndoのZNE・PEC、Huggins/Koczorの観測量推定型VD/ESDの記述と整合 |
| C40 | SS〈内容〉 | **条件付きSS**。Emerson p.1は平均ゲート忠実度を支持。一般のdephasing率同定までは支持しない |
| C43 | SS | **SS**。PQES v2 abstractおよび§V、p.9。ただし閾値の適用条件は保持すべき |
| C44 | PS | **PS**。Unified §6.13・Table 14、p.14。触媒保存の検査は存在するが結果は保存失敗 |

無作為抽出分では、**監査表が付した限定を含めれば、アクセスできた11件の主ラベルは維持できる**。

一方、追加確認した次の箇所は修正が必要である。

- **C12**：主張要約が「exact **iff** target full rank」となっている。原典は **if**。恒等変換なら純粋状態を純粋状態へ正確に変換できるので、「iff」は偽。主張全体へのSSは不適切で、少なくともPSにするか、要約を訂正すべき。
- **C20**：引用されたMarvianの命題自体はSS。しかし、これを今回の**触媒付き・成分単位・閾値付きマスク**の正当化に使う部分までSSとはできない。主張全体ならPSが妥当。

**反証条件：** 固定seedの抽出結果が異なる、または上記11件について示した一次資料が実際には限定後の主張を支持しないことが確認されれば、私の抽出・再判定・90点の採点は誤りである。

## B. 総合確度：80/100

対象別には、**引用監査表は90/100、修正後原稿は75/100程度**と評価する。総合点は単純な誤字数ではなく、中心的な説明と実装の一致を重く見た。

最も確度を下げる具体的事実は次の3件。

1. **証明対象とTable 2の生成写像が異なる。**  
   \(\rho_e=\rho_t,\Delta_{\rm mode}=0\) なら定義上 \(F=1\) なのに、表生成経路は追加補間によって0.955等を返す。この不一致を直接再現した。

2. **Sec. 5.10のQEM比較は、他の節で重要だとするモード制限を省略している。**  
   同一の推定状態でも、\(d=64\) では約1.000から0.293345へ変わる。「同じBlind CQECの性能」として横断的に読むことはできない。

3. **閾値による切捨てを、物理的なno-goや性能上限として説明している。**  
   Regev・dephasing \(\gamma=2\) の再計算では、非ゼロの非対角成分が**380個**残った。最大は \(2.3033\times10^{-12}\)、最小は \(1.8337\times10^{-69}\)。これらは \(10^{-10}\) 未満だが、数学的なゼロではない。  
   また、前掲3×3例では \(E=(0,1,2)\) に対して入力・標的とも \(C=\mathbb Z\)。Shiraishiの十分条件は任意精度の触媒変換を許すが、当該インターフェースの忠実度は **0.686887**である。したがって、その値を一般的な物理実装の上限とは呼べない。

## C. 確認した誤認・不整合

- **`paper_blind_cqec_qst.tex:162`** — 全シミュレーションの回復は \(\Pi\circ\mathcal M\) → **少なくともアルゴリズム表では、その後に有限コピー補間を行う別写像がある**。写像を明記して誤差項を追加するか、表を再計算する必要がある。出典：`benchmark_blind_algorithms.py:193`、`benchmark_blind_algorithms.py:197`、`icec.py:303`。

- **`paper_blind_cqec_qst.tex:800`** — 定義されたBlind ChInvが全次元で1.000 → **当該比較の実装はモード制限なしの逆チャネル処理**。同一標本へ定義どおりの回復を適用すると、\(d=64\) の平均は0.293345。出典：`benchmark_blind_qem_compare.py:115`、数値再実行。

- **`paper_blind_cqec_qst.tex:506`** — 有限dephasingで「Theorem 1 forbids any recovery」 → **数値閾値により全対象成分を切り捨てた結果であり、Theorem 1による禁止ではない**。出典：再計算した380個の非ゼロ成分、Shiraishi本文pp.2–4。

- **`paper_blind_cqec_qst.tex:133`** — 全報告忠実度は物理実装の上限 → **その上限性は成立が示されていない**。保守的マスクの出力より高い忠実度が存在定理で許される。出典：前掲3×3反例、Shiraishi Theorem 2、p.3。

- **`paper_blind_cqec_qst.tex:170`** — \(\mathcal M\) は不在のgapを削除する → **実装は各行列成分の閾値マスク**。同じgapを持つ別成分や、整数結合で到達可能なgapとの区別が必要。またPSD射影後のゼロ保存も保証されない。出典：同原稿169行、`blind_cqec_pkg/blind_cqec/recovery.py`、前掲反例。

- **`paper_blind_cqec_qst.tex:394`** — 四つの具体例はUnifiedの例 → **QKANは少なくとも異なる**。Unifiedは \(x=0.5\) のChebyshev値・次数打切り、今回のコードはsin近似。出典：Unified §5.1、p.8、`benchmark_blind_algorithms.py:315`、`benchmark_icec.py:425`。

- **`paper_blind_cqec_qst.bib:266`** — `Harrow, Arllen W.` → **`Harrow, Aram W.`**。出典：Haah PDF、p.1著者欄。

- **`paper_blind_cqec_qst.bib:79`／`:83`** — v2本文を監査した引用が `2603.11568v1` を指す → **監査した版を参照するなら `2603.11568v2` に統一**。出典：取得済みPQES PDF、p.1の版表示。

- **C08** — Shiraishi Def. S.2「補遺p.2」 → **補遺p.3**。共変性の等式への直接参照なら **Proposition S.3・(3)**。出典：PDF通し10頁。

- **C12** — exact **iff** target full rank → exact **if** target full rank。出典：Shiraishi Theorem 2、本文p.3、Theorem S.24、補遺p.15。純粋状態への恒等変換が「only if」の反例。

- **C27の独立測定下限注記** — \(\Omega(dr^2/\delta^2)\) → **対数因子を保持すれば \(\Omega(dr^2/[\delta^2\ln(1/\delta)])\)**、または明示的なsoft-\(\Omega\) 表記。出典：Haah abstract、Theorem 4、p.8。

- **C45** — rs-9366084「12 Apr 2026」 → **掲載日は2026年4月13日**。本文原稿日は別に2026年4月9日と表示される。出典：取得済みPDF表紙・本文冒頭。

**未検証として区別する項目**

- Nielsen–Chuang書籍、Shor 1995全文、Davies 1988全文は**検証不能**。Daviesについてはローカルの出版社abstractのみ確認した。
- 追加7文献の原典全面照合、39件すべてのDOI・出版社版書誌の独立照合、Clintonの2026年掲載巻号・論文番号は未検証。
- PDF再コンパイル、全8スクリプトの描画・保存を含む再実行は未実施。確認したのは既存PDF・ログと、上記の実行済み数値経路。
- 元の「hygiene grep」の検索式は不明。ここで明示した検索式の0件という結果とは区別する。

**未検証項目を偽とは判定していない。主要な減点は、実際に再現できた写像の不一致と、閾値切捨てを物理的制約に読み替えた点による。**

## Claude Round 1
## A. カテゴリ別確度

### (1) 理論系引用の位置・番号 — **92**

一次PDFを直接開いて全件照合した（`/Users/deeptell01/research/raw/`）。

| 主張 | 検証結果 |
|---|---|
| Watrous Eq.(1.173) p.33 = Schatten双対 | ✓ 一致（PDF p.41 = 刷り p.33） |
| Watrous Eq.(1.182) p.34 = `‖X‖₁=max{|⟨U,X⟩|}` | ✓ 逐語一致 |
| 「Thm 1.14 は Jensen の不等式、p.50」 | ✓ 正しい（PDF p.58、巻末索引 "Jensen's inequality, 50"） |
| Sec. 1.1.3 は刷り p.24–34 → (1.173)(1.182) は同節内 | ✓ 目次で確認 |
| Jordan–Hahn Eqs.(1.143)–(1.144), pp.26–27 | ✓ 索引 "Jordan–Hahn decomposition, 26–27" と一致 |
| FvdG Thm 1 Eq.(46) p.17 / B は Prop.4 Eq.(24) p.13 | ✓ 逐語一致。`B = Tr√(√ρ₀ρ₁√ρ₀)` = 非二乗忠実度 → 原稿 N5 の「B=√F」は整合 |
| ZS Sec. II.C Eq.(2.8) p.3、Eq.(2.9) p.4、App. A p.12 | ✓ 一致。モーメントは**確かに未記載** |
| Shiraishi Thm 2 (本文 p.3) / Thm S.24 (補遺 p.15) | ✓ 逐語一致。full-rank 条件は**ターゲット側**で正しい |
| Shiraishi Def. S.5 / S.6（補遺 p.4）、Eqs.(4)–(5) | ✓ 一致 |
| Shiraishi Thm 3（本文 p.4、C′= 有理スパン、converse は conjecture） | ✓ 一致 |
| Marvian Prop.1 Eq.(2.11) p.6 / Eq.(2.5) p.5 | ✓ 逐語一致 |
| Haah Table I p.2 / Thm 3・Thm 4 p.7 | ✓ 一致 |
| Tropp Thm 1.6.2 (Matrix Bernstein) p.8 | ✓ 一致 |
| Davies 1988 抄録 | ✓ 逐語一致（"…false for p=1 and p=∞, but results only marginally (logarithmically) weaker do hold in finite dimensions"） |

**唯一の誤り（監査側）**: C08 の「time-translation 形は Shiraishi **Def. S.2**（補遺 **p.2**）」。実際は **Proposition S.3**（補遺 **p.3**）。Def. S.2 はエネルギー保存ユニタリによる Stinespring 形で、`Λ∘U_t = U_t∘Λ` は Prop. S.3。原稿本体は番号を付けずに引用しているため伝播はしていない。

**反証条件**: Watrous 第2版で Thm 1.14 が Jensen 以外であれば／Shiraishi 補遺の PDF ページラベルが実際の刷りページとずれており Def. S.2 が p.2 に載る版が存在すれば、私の採点は誤り。

### (2) 応用系引用の事実 — **76**

| 主張 | 検証結果 |
|---|---|
| rs-9366084 実タイトル = "Biologically Plausible QEC in a Three-Layer Neural Spin Model", Wakaura 単著 | ✓ PDF 表紙で確認。bib も修正済み |
| rs-9366084 投稿日「12 Apr 2026」 | ✗ **PDF は "Posted Date: April 13th, 2026"**（監査の1日ずれ） |
| Clinton 著者6名 | ✓ PDF と bib 完全一致。ただし **bib に arXiv id なし**（PRX Quantum 7, 010345 表記のみ） |
| PQES v2 タイトル・閾値 3/4, 1/2 | ✓ PDF 抄録に `p_th = 3/4`（depol）, `1/2`（deph）。PDF 自体の刻印は **arXiv:2603.11568v2** |
| PQES bib | ✗ `eprint = {2603.11568v1}` / `url = .../abs/2603.11568v1` が **v1 のまま**（監査の是正 #16 が半分だけ適用） |
| Unified に "ICEC" 不在 | ✓ **大文字 ICEC = 0 件**、`icec.py` のみ p.5・p.19 |
| `F_cat^marg` 0.48–0.84 | ✓ Table 14 (p.14) 実値 0.480/0.569/0.756/0.764/0.844 |
| Unified Sec. 2.3 p.3「infinite reuse」/ Sec. 8 p.18「consumed rather than preserved」 | ✓ 逐語一致 |
| Unified Sec. 7.4 p.17 因数分解の循環性 | ✓ 逐語一致、原稿 Sec. 4.1 に反映済み |
| Emerson 2005 に "randomized benchmarking" | ✓ **全文 0 件**。"dephasing" も **0 件**。"average gate fidelity" 1 件（抄録） |
| Ivashkov に "sin(πx)" | ✓ **0 件**（"sin(x)" が p.10 に例示として1回のみ）。degree-3 は Fig.10 p.18 のガウシアン準備 |
| Regev に "N = 15" | ✓ 0 件 |

**未解決の重大不一致**: 原稿 Sec. 4.1 は「具体的インスタンスは Ref.[Wakaura2026unified, Sec. 5] のベンチマークスイートのもの」と明記した上で QKAN を「target function **sin(πx)**」とする。しかし Unified Sec. 5.1 (p.8) は逐語で「the first four Chebyshev polynomials `T_n(x)` at `x = 0.5`: amplitudes ∝ (1.0, 0.5, −0.5, −1.0)」。監査自身が C30 と是正項目 #10 で「reconcile」を要求していたが**未適用**。コード側 (`benchmark_icec.py:424` `target = np.sin(pi*x)`) は sin なので、誤っているのは Unified への帰属文。同様に CF-QPE の「3-site」も Unified には記載がない（Unified は "Fermi–Hubbard-type", 16次元とのみ）。

**反証条件**: Unified の別バージョン（v1–v3）の Sec. 5.1 が sin(πx) を記載していれば／rs-9366084 の Research Square 表示日が v1 で 4/12 であれば、私の採点は誤り。

### (3) 数学的主張 — **97**

すべて自分で再計算した（`scratchpad/verify_math.py`）。

- **Lemma 1**: 証明を段階ごとに検算。(L2) `Π(A)−A = A₋ − (s/(1+s))A₊`、直交台より `‖·‖₁ = s + s = 2s` ✓。(L3) `Y = I−2P₋` は `−I ≤ Y ≤ I`、`tr[Y(A−σ)] = 2s + 2tr[P₋σ] ≥ 2s` ✓。数値検証: ランダム Hermitian trace-one × 密度行列 200,000 組（d=2–5）で**違反 0 件**、達成最大比 **1.0083**（係数2は緩い上界だが妥当）。
- **Theorem 2**: (T1)–(T6) の連鎖に論理的欠落なし。`Δ_mode=0 ⇒ A=ρ̂_est` は `‖M(ρ̂)−ρ̂‖₁=0` から従う ✓。
- **Corollary 1**: `‖X‖₁ ≤ √d‖X‖₂` → Jensen → 独立性で `√(d/n)`、`O(n^{-1/2})` ✓。
- **「二側 clipping 不等式は偽」**: `‖A₊−B₊‖₁/‖A−B‖₁` をランダム探索。**d=2 では最大ちょうど 1.0000（違反なし）、d=3 で 1.0241 の反例を取得** → 3×3 が最小反例次元という原稿の主張と整合。反例行列も出力済み。
- **Dirichlet モーメント**: Haar 純粋状態 400,000 サンプル× d=2,3,4,8,16,64。`E[p_i]=1/d`, `Var=(d−1)/(d²(d+1))`, `E[p_ip_j]=1/(d(d+1))` すべて 3–4 桁一致（例 d=64: Var 2.341e−4 vs 理論 2.366e−4）。
- **M が PSD を保存しない**: `|+++⟩⟨+++|/3` の (0,2) 成分をゼロ化 → 最小固有値 **−0.1381** ✓。
- **符号規約**: `U_t = e^{−iHt}(·)e^{iHt}` から `[U_t(ρ)]_{ij} = e^{−i(E_i−E_j)t}ρ_{ij}` → 原稿の `e^{−iωt}` は自己整合、かつ Shiraishi 補遺 p.3 の `ρ_ij(t) = e^{i(E_j−E_i)t}ρ_ij` と一致。Marvian の `e^{+ikθ}` からの符号反転を「generalizing」と明示した処理も正しい。

**反証条件**: Lemma 1 の反例（`‖Π(A)−σ‖₁ > 2‖A−σ‖₁`）が1つでも構成されれば／2×2 で二側 clipping 比 > 1 が示されれば、私の採点は誤り。

### (4) 数値・コード主張 — **84**

- **91 tests pass**: `pytest -q` → `72 + 19 = 91`、exit 0 ✓
- **results/*.json 8件** ✓（algorithms, circuit_sanity, cqec, dimension_sweep, nature_supp, qem_compare, sensitivity, vqe_demo）
- **36 頁**: PyMuPDF で直接 36 ✓（.log も "36 pages"）。PDF (20:11) は .tex (20:03) より新しく最新
- **「ICEC」0件** ✓
- **Table 2**: 監査が挙げた4値 `0.955 / 0.986 / 0.956 / 0.738` は JSON と**完全一致** ✓。ただし同表の他2セルに丸め誤り（下記 C 参照）
- **Regev 天井の「44%」**: `coherence_ratio = 0.4432` ✓
- **QEM 比較**: ZNE 0.513 / VD 0.670 (d=4)、0.031 / 0.041 (d=64)、CoM 0.294 (d=64)、PEC・ChInv 1.000 全次元 — すべて JSON 平均と一致 ✓
- **感度**: d=64 で 0.52→0.20 (−5%), 0.12 (−10%), 0.34 (+10%)、キャプションの 61%/35% も再計算一致 ✓。γ 単独 −10% → 0.158、p ±30% → 0.493–0.519、γ_AD −30% → 0.222 ✓
- **混合状態 v=0.3**: CoM 0.783 vs 無補正 0.921 ✓
- **VQE 3.4×**: `abs_error` 0.3387 → 0.1004、比 3.373 ✓
- **相関**: `n_points = 84`（「84-condition」✓）、`per_dim_r` d=64 = 0.770 ✓、d=4/8 = 0.9988/0.9978（「r ≥ 0.998 at d ≤ 8」は d=8 で 0.99785、3桁丸めで辛うじて成立）
- **hygiene grep 0件**: ✗ CLAUDE.md 指定の grep は **5件ヒット**（行 56, 446, 655, 748, 855）。ただし全て "corrected expectation values" / "uncorrected state" 等の技術的用法で、規則が禁じる版比較語ではない。**「0 hits」という記述自体は事実に反する**
- **REPRODUCE.md**: 対応表は JSON のトップレベルキーと一致 ✓。ただし本文が `paper_blind_cqec_quantum.tex`（`_qst.tex` ではない）を参照

**反証条件**: 別環境で pytest が 91 以外になる／`benchmark_blind_algorithms.py` を再実行して Table 2 の Iterative・Multi-copy 列が 0.854 / 0.048 を返せば、私の採点は誤り。

### (5) 監査ラベルの妥当性 — **91**（Status 節のみ **55**）

10件の要求に対し **23件**（C02, C03, C08–C12, C14, C16, C17, C19, C21–C24, C27, C30–C32, C34, C36, C39–C41, C43–C45）を一次資料で再判定。**ラベル（SS/PS/CT/NF）の妥当性は全件支持**。特に C26（Gross を CT）と C45（rs-9366084 を CT）は正しい判断。誤ラベルは0件、位置誤りが2件（C08、C45 の日付）。

一方 **Status 節の3つの自己申告は不正確**: 「All 18 corrections applied」（#10 の QKAN 整合と #16 の PQES バージョンが未/半適用）、「0 hygiene hits」（実測5件）。

**反証条件**: 未検証群（下記）を開いた結果、SS/PS ラベルの誤りが3件以上見つかれば、私の採点は誤り。

### 未検証項目（確度計算から分離）

ローカルに一次資料がない、または今回開いていないもの — **確度を下げる根拠にはしていない**:
Nielsen–Chuang（書籍、C01/C15/C18/C33）、Shor 1995 全文、Davies 1988 本文（抄録のみ検証）、Jozsa 1994 本文、Leggett 1987 本文、Breuer–Petruccione、Knill 2005「above 3% per gate」、Hradil 1997、Gross 2010 抄録・Thm 1、Temme/Li/Endo/Koczor/Huggins の抄録引用（C37）、Chen 2021 Sec. II.D、Clinton Sec. 2.1.2、Regev Sec. 3、Gour–Spekkens Eq.(15)、O'Donnell–Wright、Campbell 2019、Knill 2008、Magesan 2011、Baumgratz 2014。

## B. 総合確度: **82**

最も確度を下げている具体的事実3件:

1. **原稿 971行の「5--10 copies at d ≤ 16」が自前データと矛盾**。同じ原稿の Table 4（`tab:min_copies`）は同条件で 50–200 回、Abstract/Introduction も「n ≈ 50–200」。JSON の copy grid `[2,5,10,20,50,100,200]` で n=5–10 における最良値は 0.52–0.82 に過ぎず、閾値 F_rec ≥ 0.95 に遠く届かない。監査が \cite を持つ文だけを対象にしたため完全に見落としている。

2. **QKAN インスタンスの帰属が一次資料と食い違ったまま**。Sec. 4.1 が「インスタンスは Unified Sec. 5 のもの」と明言しつつ sin(πx) と書くが、Unified Sec. 5.1 は Chebyshev T₀–T₃ at x=0.5。監査 C30／是正 #10 が指摘済みなのに未適用で、監査は「全18件適用」と申告している。

3. **監査 Status 節の自己申告が2点で事実と不一致**（「0 hygiene hits」実測5件、「全18件適用」→ #10 未適用・#16 半適用）。監査本体の品質は高いのに、完了報告が検証を通らない。

## C. 発見した事実誤認

| 箇所 | 誤 → 正 | 出典 |
|---|---|---|
| `paper_blind_cqec_qst.tex:971` | 「blind CQEC requires as few as **5--10 copies** at d ≤ 16」→ 同原稿 Table 4 に従い **50--200**（あるいは判定基準を明示して書き直す） | `blind_cqec_pkg/results/benchmark_blind_algorithms.json` → `tab_min_copies`（50/100/200/>200）および `fig_blind_alg_copies`（n=10 で最良 0.817） |
| `paper_blind_cqec_qst.tex:391-397` | QKAN インスタンスを Unified Sec. 5 に帰属しつつ「target function **sin(πx)**」→ Unified の当該インスタンスは **Chebyshev T₀–T₃ at x = 0.5, 振幅 ∝ (1.0, 0.5, −0.5, −1.0)**。sin(πx) は本研究独自の設定として切り分けるべき | `catalytic-quantum-error-correction.pdf` Sec. 5.1, p.8（逐語） |
| `paper_blind_cqec_qst.tex:484`（Table 2, CF-QPE 行 Iter. 列） | **0.854** → **0.853** | JSON 実値 `0.8534910623114137` |
| `paper_blind_cqec_qst.tex:485`（Table 2, Regev 行 Multi-c. 列） | **0.048** → **0.047** | JSON 実値 `0.04747765425243958` |
| `paper_blind_cqec_qst.bib:Haah2017` | author `Harrow, **Arllen** W.` → `Harrow, **Aram** W.` | `sample-optimal-tomography-of-quantum-states.pdf` 表紙 |
| `paper_blind_cqec_qst.bib:Raghoonanan2026` | `eprint = {2603.11568**v1**}`, `url = .../abs/2603.11568**v1**` → **v2**（引用している閾値・タイトル・"full outcome record" は全て v2 の内容） | PDF 自己参照刻印 `arXiv:2603.11568v2`、Dated July 31, 2026 |
| `CITATION_AUDIT.md` C08 | 「time-translation 形は Shiraishi **Def. S.2**（Suppl. **p.2**）」→ **Proposition S.3**（Suppl. **p.3**）。Def. S.2 はエネルギー保存ユニタリ形 | Shiraishi 補遺 p.3（PDF p.10）: "Proposition S.3. …if and only if Λ(e^{−iH_S t}ρe^{iH_S t}) = e^{−iH_{S′}t}Λ(ρ)e^{iH_{S′}t}" |
| `CITATION_AUDIT.md` C45 | rs-9366084「(**12** Apr 2026)」→ **13 Apr 2026** | `biologically-plausible-quantum-error-correctionin.pdf` 表紙 "Posted Date: April 13th, 2026" |
| `CITATION_AUDIT.md` Status 節 | 「0 hygiene hits」→ CLAUDE.md 指定 grep は **5件**（56, 446, 655, 748, 855 行。いずれも良性の技術用語だが「0」は不正確） | 実行結果 |
| `CITATION_AUDIT.md` Status 節 | 「All 18 corrections applied」→ **#10 の QKAN 整合は未適用、#16 の PQES バージョン更新はタイトルのみで eprint/url は v1 のまま** | 上記2項 |
| `blind_cqec_pkg/REPRODUCE.md` | 図表対応表が `paper_blind_cqec_**quantum**.tex` を参照 → 現行原稿は `paper_blind_cqec_**qst**.tex` | ファイル名照合 |

軽微な観察（誤認とまでは言わない）: Sec. 4.1 の「3-site Fermi–Hubbard (d = 16)」は、コード上 `n_sites=3 → n_qubits=3 → 8次元`であり、d=16 は時系列サンプル数 `N_times=16` に由来する符号化次元。物理的には 3 サイト系の Hilbert 空間ではないため、読者を誤導しうる。また Unified Sec. 5.1 には「3-site」の記載はない（"Fermi–Hubbard-type", 16次元とのみ）。
## Codex Round 2+3
## 1. 相互批判：Claude の「数値・コード84」は過大評価である

以下のファイル参照は、特記しない限り `/Users/deeptell01/Documents/alterego/personal/cqec/` 基準。今回は対象ファイルを再読し、描画・保存処理を除いた実際の計算関数をメモリ上で実行した。

### 1.1 「JSONと一致」を「定義した手法の検証」と取り違えている

Claude は次の二点を根拠に数値・コードを84点としている。

> 「QEM 比較……PEC・ChInv 1.000 全次元 — すべて JSON 平均と一致」  
> 「Table 2……JSON と完全一致」

**数値の転記一致は正しい。しかし、計算された写像が原稿の定義と一致していない。** これは丸め誤りより重大である。

`paper_blind_cqec_qst.tex:162` は、全シミュレーションの回復を
\[
\rho_r=\Pi\!\left(\mathcal M_{\rho_n}(\widehat\rho_{\rm est})\right)
\]
と定義する。一方、`benchmark_blind_qem_compare.py:115` は
```python
def icec_recover(noisy, target_est):
    return psd_project(target_est)
```
であり、**`noisy`を使わず、\(\mathcal M\)を実行しない。**

当該スクリプトの関数をそのまま読み込み、`seed=42`、次元順 `[4,8,16,32,64]`、各15状態で再計算した。同じ推定状態に対し、回復部分だけをパッケージの定義に置き換えると：

| 次元 | QEMスクリプトの平均忠実度 | \(\Pi\circ\mathcal M\)による平均忠実度 |
|---|---:|---:|
| 32 | 約1.000000 | **0.592606297** |
| 64 | 約1.000000 | **0.293345095** |

したがって、**「定義されたBlind CQECが全次元で1.000を達成する」という読みは、この入力で失敗する。** 後者のマスクを物理的に正しいと認定したわけではない。ここで証明したのは、比較対象の同一性が崩れていることだ。

### 1.2 Table 2のコピー依存性には、定理とは別の補間が入っている

`benchmark_blind_algorithms.py:193` はモード制限付き回復を実行するが、続く197行から `ICECProtocol.correct` を追加実行する。その内部、`icec.py:303` には
\[
c_n=1-\frac1{1+0.1n},\qquad
\rho_{\rm out}=c_n\rho_{\rm restricted}+(1-c_n)\rho_{\rm amplified}
\]
という補間がある。

**推定状態を真の標的に固定**し、同じアルゴリズム状態・combined noise・\(n=100\)で比較した。

| 入力 | 原稿の \(\Pi\circ\mathcal M\) | 実際のOracle表生成経路 |
|---|---:|---:|
| qDRIFT | **1.000000** | 0.955054129 |
| QKAN | **1.000000** | 0.985856486 |
| Control-free QPE | **1.000000** | 0.956427970 |

最初の三例は推定誤差もマスクによる損失もゼロなのに、表生成経路では忠実度が低下する。**Theorem 2への反例ではなく、「全シミュレーションが同じ回復式」という説明への反例**である。

さらに、Claude が確認したという「Regev天井」も、固定された構造的上限としては成立しない。保存JSONでは、同じcombined noise・Channel inversionで
\[
F(n=100)=0.738349784,\qquad
F(n=200)=0.754667021.
\]
`paper_blind_cqec_qst.tex:494` の「0.738は構造的限界」という説明に対し、**同じ実装で入力コピー数を200にすると、その値を超える。** `coherence_ratio=0.4432`との数値照合だけでは、この上限解釈を検証したことにならない。

### 1.3 「閾値切捨て＝no-go」と「射影後も消去が維持される」を検証していない

`paper_blind_cqec_qst.tex:506` は、Regev・dephasing \(\gamma=2\)について「Theorem 1 forbids any recovery」と書く。再実行では、非対角成分が**380個、非ゼロのまま**残る。

\[
\min|\rho_{ij}|=1.83374418\times10^{-69},\qquad
\max|\rho_{ij}|=2.30333284\times10^{-12}.
\]

全て \(10^{-10}\)未満だが、数学的なゼロではない。有限のdephasingでは、元の非ゼロ成分に正の指数因子が掛かっている。**数値マスクが回復を打ち切ったことは再現できるが、物理的禁止は導けない。** Shiraishi–Takagi自身も、微小な非ゼロコヒーレンスと厳密なゼロを区別している。原稿のTheorem 1はそもそも十分条件であり、その条件を数値的に不成立と判定しても禁止命題にはならない。([arxiv.org](https://arxiv.org/abs/2308.12338))

また、実装の「消した成分は回復状態でもゼロ」という説明も、次の入力で失敗する。
\[
\rho_n=
\begin{pmatrix}
1/3&0.1&0\\
0.1&1/3&0.1\\
0&0.1&1/3
\end{pmatrix},\qquad
\widehat\rho_{\rm est}=\frac13
\begin{pmatrix}1&1&1\\1&1&1\\1&1&1\end{pmatrix}.
\]

パッケージの `icec_recover` を実行すると、
\[
\lambda_{\min}(\mathcal M(\widehat\rho_{\rm est}))
=-0.1380711875,\qquad
[\Pi(\mathcal M(\widehat\rho_{\rm est}))]_{02}
=\boxed{0.03033008589}.
\]
**マスクが消した成分を、PSD射影が再生成する。** 該当説明は `blind_cqec_pkg/blind_cqec/recovery.py:37`。

ただし、これを「禁じられた物理モードを生成した」と言い換えてもいけない。\(E=(0,1,2)\)なら入力・標的の共鳴モード格子は共に \(\mathbb Z\)であり、gap 2は整数結合で到達可能である。この例が示すのは、**成分マスク・周波数モード・共鳴格子は別物**だということである。

同例の出力忠実度は **0.686886724**。一方、モード包含の十分条件は任意精度の触媒変換を許すため、この数値を一般の物理実装の忠実度上限とはできない。`paper_blind_cqec_qst.tex:133` の「全報告忠実度は物理実装の上限」は、少なくとも上限を取る対象と資源条件を限定し直す必要がある。

### 1.4 「監査ラベルは全件支持」も、C12で成立しない

Claude は、

> 「ラベル（SS/PS/CT/NF）の妥当性は全件支持」

とする。しかし `CITATION_AUDIT.md:24` のC12は、監査対象の主張を **“exact iff target full rank”** と書き、SSを付けている。原典のTheorem 2、本文p.3が述べるのは **full rankならexactという十分条件**であり、必要条件ではない。

反例は
\[
\rho=\rho'=|+\rangle\langle+|,\quad
\Lambda=\mathrm{id},\quad \dim C=1.
\]
標的のrankは1だが、変換は厳密で触媒も保存される。実行結果は標的rank 1、変換誤差0、時間発展との共変性誤差0だった。**したがって、C12を記載された主張全体に対してSSとする判定には同意できない。**

加えて、Claude の「23件」とする列挙は展開すると**27件**である。C26を支持したとも書くが、その列挙には含まれず、Gross本文は未検証群にも入っている。これは誤ラベルの追加証明ではないが、検証範囲の報告は訂正が必要である。

## 2. 正当なので受け入れる指摘

指定された五項目は、**いずれも対象ファイルに存在する事実**だった。ただし、意味と重大度は区別する。

| Claudeの指摘 | 再検証と譲歩 |
|---|---|
| L971「5--10 copies」 | **存在する。** `paper_blind_cqec_qst.tex:971`。\(F_{\rm rec}\ge0.95\)を意味するなら自前データが支持しない。私のRound 1の明示的な見落としとして受け入れる。 |
| Table 2：0.854→0.853 | **正しい。** `paper_blind_cqec_qst.tex:484`。JSON実値は0.8534910623114137で、通常の小数3桁丸めは0.853。 |
| Table 2：0.048→0.047 | **正しい。** `paper_blind_cqec_qst.tex:485`。JSON実値は0.04747765425243958で、丸めは0.047。 |
| Haah著者「Arllen」 | **正しい指摘。** `paper_blind_cqec_qst.bib:266`。取得済みHaah PDF表紙は **Aram W. Harrow**。 |
| PQES bibがv1 | **正しい指摘。** `paper_blind_cqec_qst.bib:79` と `paper_blind_cqec_qst.bib:83` はv1。監査対象のローカルPDFはv2。 |
| hygiene grep 5件 | **正しい。** 親ディレクトリの `CLAUDE.md:178` にある式で再実行し、56、446、655、748、855行の5件。私のRound 1の別検索式による0件では、この主張を検証できていなかった。 |

**ただし、「5–10 copiesはデータと矛盾する」の断定には達成基準が必要である。** Channel inversionの保存値は：

| アルゴリズム | \(n=5\) | \(n=10\) | 探索グリッドで初めて \(F\ge0.95\) |
|---|---:|---:|---:|
| qDRIFT | 0.520991 | 0.669888 | 100 |
| QKAN | 0.693190 | 0.816952 | 50 |
| Control-free QPE | 0.565555 | 0.698471 | 100 |

5–10コピーでも無補正からの改善はある。したがって正確な批判は、**「高忠実度達成を含意する資源比較として不適切」**であって「改善すらしない」ではない。また、50–200という代替値も、前述の補間モデル上の値であり、物理的な測定・触媒準備のコピー費用を実証した値ではない。

hygieneの5件も良性の技術用語である。「0 hits」という完了報告は誤りだが、**科学的な主要欠陥の上位三件に置くのは重み付けを誤っている。**

QKAN帰属の批判は受け入れる。ローカルUnified PDFの§5.1、p.8はChebyshev値 \(T_0,\ldots,T_3\) at \(x=0.5\)、今回の実装はsin近似であり、同一インスタンスへの帰属は不正確である。

なおPQESの日付は、PDFの版刻印が**2026年7月30日**、本文の“Dated”が**2026年7月31日**。私とClaudeの記述は参照した日付欄が異なり、これだけで相互矛盾とはならない。

## 3. 最終立場と確度

### 82対80の差は何だったか

Round 1の「私−Claude」のカテゴリ差は順に **+3、+9、−7、−14、−1**。最大の不一致は数値・コードであり、私は**写像と物理解釈の不一致**を、Claudeは主に**保存値との一致**を評価していた。

ただし単純平均は私86、Claude88で、報告した総合80・82とは異なる。両者とも総合の集約規則を明示していなかったため、**総合2点差を精密な確率差として正当化することはできない。**

今回は集約を明示する。中心的な数値結論への影響を重視し、カテゴリ重みを順に **10%、10%、25%、45%、10%**とする。以下は統計的に校正された確率ではなく、反証可能な証拠に基づく審査上の確度である。

| カテゴリ | 最終確度／Round 1 | 理由 | 具体的反証条件 |
|---|---:|---|---|
| **(1) 理論系引用の位置・番号** | **95／95** | 主要位置の一致は維持。C08とHaahの転記精度の問題も維持。 | 同じ取得済み版で、Def. S.2が補遺p.2に存在する、またはHaah Thm 4の該当下限に指摘した対数因子がないと確認されれば、該当判定を撤回する。 |
| **(2) 応用系引用・書誌** | **85／85** | QKAN帰属、PQES版、著者名・掲載日の不整合は確認済み。既にRound 1で減点した項目が中心。 | 監査対象と同じUnified版の§5.1が今回のsinインスタンスを記載し、対象bibが実際にはv2・Aramを指していると確認されれば、減点を撤回する。 |
| **(3) 数学的主張** | **90／90** | Lemma 1・Theorem 2の代数的証明は支持。ただしマスクの物理解釈・上限性まで正当化したとは認めない。 | 提示した3×3行列でclipping差が0.1以下になる、射影後の(0,2)成分がゼロになる、またはLemma/Theoremの仮定下で不等式違反が再現されれば、判定を改める。 |
| **(4) 数値・コード** | **65／70** | コピー資源の追加不整合と丸め誤りを確認。写像の不一致が主要減点であり、84は支持しない。 | 同じQEM入力でモード制限を実行してもd=64平均が1.000となり、かつTable 2の実行経路に追加補間が存在しないと確認されれば、主要減点を撤回する。 |
| **(5) 監査ラベル** | **88／90** | C12の“iff”を単なる表記問題でなく、SS判定の過大評価として扱い直す。 | 監査原文が“if”である、またはrank-1標的への恒等変換がC12の対象条件から明示的に除外されているなら、C12による減点を撤回する。 |

**総合確度：79／100（Round 1：80）**。今回の明示した重みでは78.55を四捨五入した値であり、以前の80を同じ計算式から再構成したものではない。

総合評価の主要反証条件は、**原稿・全ベンチマークの回復写像が一致し、コピー数を物理的費用に結び付ける根拠と、報告忠実度の上限性の証明が提示されること**。それが満たされれば、現在の最大減点理由は解消する。

今回も **91 tests passed、JSON 8件、既存PDF 36頁**を確認した。しかしテスト合格は、原稿との意味的一致まで保証しない。Nielsen–Chuang、Shor 1995全文、Davies 1988全文などの未検証範囲は偽と判定せず、減点理由から分離する。

## 4. 投稿可能にするために必須の修正：重要度順、最大8件

1. **no-goと性能上限の誤解釈を除く。** 数値閾値・厳密なゼロ・周波数モード・整数／有理スパンを分離し、L506の禁止命題、0.738の構造的天井、一般の物理実装に対する上限という記述を撤回または証明付きで限定する。

2. **全ベンチマークの回復写像を明示・整合させる。** \(\Pi\circ\mathcal M\)、その後の有限コピー補間、QEMの\(\Pi\)のみという三経路を統一するか、別手法として明記する。Table 2とQEM比較を再生成し、定義と実行経路を照合する回帰テストを追加する。

3. **コピー数を物理的資源費用と混同しない。** 補間係数 \(0.1\) と収束則の根拠を示し、準備コピー・測定shots・触媒準備費用を分離する。根拠を示せなければ「有限コピー性能」ではなく補間モデルの感度分析に位置付け直す。

4. **マスクとPSD射影の保証を修正する。** 消去成分が最終出力でゼロに保たれるとは書かない。ゼロ制約を必要とするなら、その制約とPSD性を同時に満たす処理へ変更し、定理・数値も更新する。

5. **PQESとの比較条件を揃える。** L971の5–10コピーに達成忠実度とノイズ条件を付け、モデル上のコピー数と実測資源費用を区別する。PQESの閾値には対象状態族・ノイズモデルの条件を添える。

6. **具体的アルゴリズムの帰属を直す。** QKANのsin設定は本研究独自の選択として明記する。CF-QPEでは物理系の次元と時系列を符号化した16次元を区別する。

7. **引用監査と書誌を訂正する。** C12の“iff”とSS判定、C08の位置、Haah下限の対数因子、Aramの綴り、PQES v2、Research Square掲載日を修正する。

8. **数値転記と完了報告を実物に合わせる。** Table 2の二セル、`REPRODUCE.md`の原稿名を直し、「全18件適用」「0 hygiene hits」を再検査結果に置き換える。最後に修正後ソースからPDFを生成し、その同一版で検証を完結させる。

## Claude Round 2+3 — see session transcript (agent ae7eff85507897f18); key numbers: overall 82→63, numerics/code 84→45; reproduced all four Codex implementation findings; added: interpolation 1-1/(1+0.1n) generates abstract copy claims; Sec 5.6 vs 5.10 print 0.519 vs 1.000 for the same quantity; tau_mode sweep 1e-10→0 lifts Regev ceiling 0.738→1.000; Pi re-creates 5.17% of pruned off-diagonal L1 weight in the Regev combined case.

## Facilitator: fixes applied after the panel (2026-09-15)

All eight convergent fixes were applied; the manuscript, package results, and both submission bundles were regenerated.

| # | Panel finding | Action |
|---|---|---|
| 1 | Table 2 / Fig. copies / Table 6 used Π∘M plus an undocumented interpolation `1−1/(1+0.1n)` | `benchmark_blind_algorithms.py` now uses the package `icec_recover` only. Table 2: ChInv = Oracle = 1.000 at d ≤ 16, 0.773 for Regev. Table 6 rewritten: single-representation strategies are n-independent; "n ≈ 50–200 copies" removed everywhere (abstract, intro, Sec. 5.4, regimes). |
| 2 | QEM script ignored the noisy state (blind ChInv 1.000 at all d) | `benchmark_blind_qem_compare.py` delegates to the package map; Sec. 5.10 now reports blind ChInv 0.593 / 0.293 at d = 32 / 64 vs PEC 1.000, and explains the gap as the interface's mode restriction. Table 7 runtimes regenerated. |
| 3 | Same quantity printed 0.519 (Sec. 5.6) vs 1.000 (Sec. 5.10) | All remaining scripts (`dimension_sweep`, `sensitivity`, `nature_supp`, `cqec`, `cqec_copies`, `vqe_demo`) switched to the package map; Haar sweep now 1.000 (d ≤ 16), 0.603, 0.297; per-channel tables, sensitivity, mixed-state, qutrit text and conclusion updated from the regenerated JSONs. |
| 4 | "Theorem 1 forbids" / "structural ceiling" is τ-dependent | Reworded to threshold language throughout; new Table 9 (τ = 1e-10 … 0) added to Sec. 7 with the new `benchmark_blind_tau_sensitivity.py`; D1 rewritten as a numerical convention. |
| 5 | Π re-creates 5.17 % of pruned off-diagonal L1 | Stated in Sec. 7 with the number, and in D1 (iii). |
| 6 | L971 "5–10 copies" contradicted Table 6 | Replaced. |
| 7 | Table 2 rounding (0.854/0.048) | Moot: table regenerated; τ table cells checked against script output (0.772 at 1e-12/1e-14). |
| 8 | Bib (Haah author "Arllen", PQES v1), audit Status overclaims, QKAN attribution | Bib fixed (Aram W. Harrow; Raghoonanan v2); CITATION_AUDIT Status errata appended; QKAN/CF-QPE items now state where the instances differ from Ref. Unified Sec. 5.1. |

Unchanged after regeneration: VQE H₂ result (0.34 → 0.10 Ha, 3.4×), LiH null, circuit-sanity table (estimation only, no recovery), hybrid w* = 1, Regev dephasing-only 0.411, mixed-state crossover (largest v with CohMax < no-correction is 0.7; text tightened from "v ≲ 0.5" to "v ≤ 0.7" because CohMax at v = 0.3 is now 0.35, not 0.78). pytest: 91 passed.
