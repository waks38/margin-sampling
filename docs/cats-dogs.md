# Gerador VAE-GAN — Cats vs Dogs

Segundo dataset da fábrica de geradores (o 1º foi raio-X). Serve como **prova de
conceito visualmente verificável** do método: qualquer pessoa julga se a
interpolação latente gato→cachorro é semântica ou virou borrão — o que é difícil
no raio-X. Os geradores aqui alimentam a comparação de técnicas de data
augmentation (rede sem DA × DA clássico × DA proposto na fronteira × outras),
avaliada por tempo de treino e performance.

## Dataset

- Fonte: competição Kaggle `dogs-vs-cats` (`/kaggle/input/competitions/dogs-vs-cats/train.zip`).
- Vem como **ZIP**; `/kaggle/input` é read-only → descompactar pro
  `/kaggle/working/dogs-vs-cats/` numa célula antes de treinar (ver notebook).
- Só o `train/` rotulado serve; o `test1/` é sem rótulo (era pra submissão) e foi
  descartado — a transição gato→cachorro exige rótulo em todos os conjuntos.
- Rótulo vem do **nome do arquivo** (`cat.N.jpg` / `dog.N.jpg`), não da pasta.
- Split **disjunto por imagem** (requisito: interseção train/val/test vazia). Não
  há conceito de paciente como no raio-X.

## Como a refatoração suporta isso (origem dirigida por config)

A camada de origem (`experiments/vae_gan/data.py`) deixou de ser hardcoded. Cada
dataset é um bloco no `config.yaml` com:

| campo        | dogs            | raio-X (default)     |
|--------------|-----------------|----------------------|
| `channels`   | 3 (RGB)         | 1 (grayscale)        |
| `classes`    | `{cat,dog}`     | `{NORMAL,PNEUMONIA}` |
| `label_from` | `filename`      | `folder`             |
| `group_by`   | `image`         | `patient`            |

O bloco também declara `experiment: vae_gan`, que faz o registry rodar a classe
`VaeGan` mesmo o bloco se chamando `dogs`. Os modelos **não mudaram** com RGB: só
a 1ª/última conv veem 3 canais; o `VGGPerceptual` já aceita RGB nativamente.

## Blocos de treino

### `dogs` — baseline (rede pequena)
`base=64`, `zdim=128`, `epochs=50`, `batch=48`. Mesma capacidade do raio-X.
Sweep: `sweep.yaml` (faixa de `beta` 0.001–0.05).

**Observação do 1º treino:** reconstrução saindo dessaturada/cinza (originais
coloridas, recon cinza). Esperado para VAE no início — recon via `mu` (média do
posterior) + `l1`/perceptual são mean-seeking, e o discriminador só entra na
época 15. Cor/nitidez tendem a subir depois disso.

### `dogs_big` — rede maior (decisão atual)
Hipótese: cats/dogs é **muito mais diverso** que raio-X (pose, raça, fundo, cor),
e o conjunto de treino é ~5x maior. A mesma capacidade espalha-se sobre um
manifold muito maior → underfit → recon borrada. A correção é **capacidade**, não
mais épocas (convergência se mede em passos; 5x mais dados já dá 5x mais updates
por época).

Mudanças vs `dogs`:
- `zdim`: 128 → **256** (dobra). Barato — só crescem as camadas FC.
- `base`: 64 → **96** (sobe pouco). Conv escala com `base²` → ~2.25x compute;
  dobrar (→128) seria ~4x e foi evitado.
- `beta` (sweep): faixa cai ~pela metade (`sweep_big.yaml`, 0.0005–0.025). O KL é
  **somado nas dims**; dobrar `zdim` ~dobra a magnitude do KL, então sem
  reescalar o `beta` as receitas ficariam em escalas diferentes.
- `epochs`: mantido em 50 **provisoriamente**. Decisão final pela curva `val_l1`
  do `dogs`: ainda descendo no fim → subir; já achatou → foi capacidade, não época.
- `batch_size`: mantido em 48. Rede mais larga = mais memória por amostra; **não**
  subir sem checar OOM na T4.

Higiene: `dogs_big` é bloco separado de propósito — os runs da rede maior não se
misturam com os do `dogs` no wandb, permitindo **comparar** capacidade pequena ×
grande (evidência pra tese).

## Como rodar (Kaggle, 2× T4)

Pré-requisitos na UI: Add Input → competição `dogs-vs-cats`; Accelerator → GPU
T4 ×2; Internet ON. Confirmar GPU: `nvidia-smi -L` e
`torch.cuda.is_available()` (se vier CPU, o acelerador está desligado — não é o
torch, que é build CUDA).

1. `git clone` + `uv sync` + chave wandb pelo secret `WANDB_API_KEY`.
2. Descompactar o `train.zip` pro `data_root` (idempotente):
   ```python
   import zipfile, os
   dst = "/kaggle/working/dogs-vs-cats"
   if not os.path.exists(f"{dst}/train"):
       with zipfile.ZipFile("/kaggle/input/competitions/dogs-vs-cats/train.zip") as z:
           z.extractall(dst)
   ```
3. `run: ["dogs"]` (ou `["dogs_big"]`) no `config.yaml`.
4. Registrar o sweep certo (`sweep.yaml` p/ `dogs`, `sweep_big.yaml` p/ `dogs_big`)
   — **sweep novo**, não reusar o id do raio-X.
5. Subir 2 agentes, 1 por GPU (`CUDA_VISIBLE_DEVICES=0/1 uv run wandb agent ...`).

`run_cap` no yaml do sweep = total de modelos (compartilhado pelos 2 agentes).
Artefatos saem nomeados pela receita (`dogs_zdim..._beta...`), separados do raio-X.

## Próximos passos

- Fechar o `dogs` (50 épocas), ler `val_l1` + grades de `interp`.
- Rodar `dogs_big`; comparar fidelidade e qualidade da interpolação vs `dogs`.
- Se `val_l1` ainda descer no fim do `dogs_big`, subir `epochs` (e só então
  reconsiderar `base`, o lever caro).
