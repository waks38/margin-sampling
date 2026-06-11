# Plano de alterações no treinamento — VAE-GAN

Notas das mudanças decididas/pendentes para o experimento `vae_gan`.

**Contexto:** gerador incondicional cujo produto é a interpolação latente
saudável→doente (*margin sampling*). A fronteira ambígua deve vir do **VAE**
(latente liso), **não** do GAN.

**Baseline atual (`config.yaml`):** 50 épocas, `beta=0.5`, batch 64, `zdim=128`,
img 128px, pesos `l1=1` / `l_perc=0.3` / `l_adv=0.7`, `beta_warm=2000`,
`disc_start=3500`.

## Ordem sugerida

1. **KL** (decisão de normalização) — destrava a qualidade do latente, base da tese.
2. **Schedule do GAN por época + checkpoint/resume + AMP** — baratos, destravam o sweep.
3. **Plumbing do sweep + higiene de dados.**

## Já feito

- **Interpolação esférica (slerp) + uso de `mu`** em `_avaliar`: o caminho fica
  na casca de raio ~√d (evita o interior vazio da corda reta) e as pontas ficam
  limpas/reprodutíveis. Recon e gate também passaram a usar `mu` → avaliação
  determinística e comparável entre épocas.
- **Normalização do KL.** Decisão: somar sobre as dims (convenção padrão,
  média só no batch) e recalibrar `beta`. O `beta=0.5` antigo equivale a
  ~`0.004` na escala nova; config foi pra `beta=0.01` como ponto de partida
  (domínio exato fica pro sweep). Runs antigos no wandb ficam incomparáveis
  em `kl`/`beta`.
- **Schedule do GAN e do warmup do KL por época.** `disc_start=3500` (passos)
  nunca dispararia (~3200 passos no total); virou `disc_start_epoch: 15` e
  `beta_warm_epochs: 10` — rampa do beta continua linear por passo, mas o alvo
  é definido em épocas. `steps_per_epoch` é logado no início do treino.
- **Checkpoint periódico + resume.** `outputs/vae_gan_ckpt.pt` a cada
  `ckpt_every` épocas: pesos + optimizers + epoch/step + estados de RNG
  (resume idêntico ao run ininterrupto) + hparams (resume falha com mensagem
  clara se o config mudou) + `wandb_id` (o chassi retoma o mesmo run nas
  curvas). VGG perceptual fica de fora (congelado). `GradScaler` entra quando
  o AMP for implementado.
- **AMP.** `autocast` + um `GradScaler` por optimizer (G e D); estados dos
  scalers entram no checkpoint. `amp: true` no config; no-op em CPU.
- **Plumbing do sweep.** `wandb.config` é a fonte da verdade após o `init`
  (overrides do agente valem sobre o yaml, incluindo `seed`); artefato nomeado
  pela receita (`vae_gan_zdim..._beta..._adv..._perc...`).
- **Higiene de dados.** Dedup no `importar` por (classe, nome, tamanho) —
  neutraliza a cópia aninhada `chest_xray/chest_xray` do zip do Kaggle.
  Resize agora preserva proporção (lado menor → `img_size`) + center crop,
  em vez de esticar pro quadrado.
- **Métrica de colapso.** KL por dimensão acumulado na época →
  `dims_ativas` (KL médio > 0.01 nat) logado toda época no wandb e no print.

## A decidir (impacta a tese)

- **GAN × meio do caminho.** Decisão: **não** alimentar o discriminador com os
  pontos interpolados. Empurrar o meio pra "parecer real" o grudaria na classe
  mais próxima (*mode-snapping*) e mataria a ambiguidade. O GAN fica só na
  reconstrução; `l_adv` é botão de trade-off nitidez ↔ suavidade. Quem julga
  "bom ponto de fronteira" é o **classificador** (confusão sobe no meio), não o
  discriminador.

## A implementar

- (vazio — falta só definir o domínio do sweep e rodar)

## Limitações conhecidas (documentadas, sem ação)

- Agrupamento por paciente é assimétrico: pneumonia agrupa por `person<id>`,
  normal vira "1 imagem = 1 paciente" (nome não tem id). Não-vazamento é
  best-effort pro normal — limitação do dado, não bug.
- O gate de "colapso" (L1 entre recons de pares não-relacionados) segue logado,
  mas o sinal primário agora é `dims_ativas`/`kl`.

## Hardware

- Rodar com torch atual numa **T4** (sm_75, suportada, AMP com tensor cores);
  **não** fixar torch antigo (cu121) só pra usar a P100 (Pascal, sem tensor
  cores, exigiria travar em torch ~2.4 e perder o AMP).
- **DataParallel (`use_dp`) deixa mais lento** neste modelo: replicar o modelo a
  cada passo + scatter/gather custa mais que o ganho com batch/imagem pequenos.
  Manter `use_dp=false`. Pra usar as 2 GPUs no **sweep**, rodar 1 agente por GPU
  (`CUDA_VISIBLE_DEVICES=0` e `=1` em paralelo) em vez de DP — dobra o throughput
  sem overhead de comunicação.
