# Plano de alterações no treinamento — VAE-GAN

Notas das mudanças decididas/pendentes para o experimento `vae_gan`.

**Contexto:** gerador incondicional cujo produto é a interpolação latente
saudável→doente (*margin sampling*). A fronteira ambígua deve vir do **VAE**
(latente liso), **não** do GAN.

## Já feito
- **Interpolação esférica (slerp) + uso de `mu`** em `_avaliar`: o caminho fica
  na casca de raio ~√d (evita o interior vazio da corda reta) e as pontas ficam
  limpas/reprodutíveis. Recon e gate também passaram a usar `mu` → avaliação
  determinística e comparável entre épocas.

## A decidir (impacta a tese)
- **Normalização do KL.** Hoje `kl_div` tira média sobre as 128 dims → o `beta`
  efetivo é ~128× mais fraco que o `beta=0.5` do config sugere → latente quase
  não regularizado. Decidir entre: somar sobre as dims (convenção padrão) e
  recalibrar `beta`, ou manter e raciocinar `beta` na escala atual. É o botão
  central da suavidade do latente.
- **GAN × meio do caminho.** Decisão: **não** alimentar o discriminador com os
  pontos interpolados. Empurrar o meio pra "parecer real" o grudaria na classe
  mais próxima (*mode-snapping*) e mataria a ambiguidade. O GAN fica só na
  reconstrução; `l_adv` é botão de trade-off nitidez ↔ suavidade. Quem julga
  "bom ponto de fronteira" é o **classificador** (confusão sobe no meio), não o
  discriminador.

## A implementar
- **Schedule do GAN por época/fração, não por passo.** `disc_start=3500` em
  passos pode nunca disparar (~3200 passos no total com batch 64) e quebra
  quando muda batch/dataset. Logar `steps_per_epoch`.
- **Checkpoint periódico + resume.** Salvar enc/dec/disc + optimizers +
  epoch/step a cada N épocas. Protege do corte de 12h do Kaggle e é
  pré-requisito pro sweep.
- **Mixed precision (AMP).** `autocast` + `GradScaler` — ~2× mais rápido e
  ~metade da memória na T4; importante pro throughput do sweep.
- **Plumbing do sweep.** `main.py` ler hiperparâmetros do `wandb.config` (não só
  do yaml) e nomear o artefato pela receita (hoje é fixo `"vae_gan"`).

## Higiene de dados (verificar)
- Conferir duplicação da pasta aninhada `chest_xray/chest_xray` (o `rglob`
  contaria cada imagem 2×).
- O resize força quadrado e distorce a proporção do raio-X; considerar resize
  preservando proporção + crop.

## Métrica
- O gate de "colapso" é fraco (pareia imagens não-relacionadas). Sinal limpo de
  colapso é o próprio `kl → 0` / nº de dims ativas. Menos urgente: com o KL
  fraco atual, colapso nem é o risco principal.

## Hardware
- Rodar em **T4 x2** com torch atual; **não** fixar torch antigo (cu121) só pra
  usar a P100. A P100 (Pascal, sem tensor cores) exigiria travar em torch ~2.4 e
  perder o AMP com tensor cores. T4 x2 = 2 GPUs (casa com `use_dp`) + versão atual.
