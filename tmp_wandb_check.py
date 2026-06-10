import glob, time
import wandb

ENT, PROJ = "felwaks-universidade-federal-de-minas-gerais", "margin-sampling"

# ---- (A) historico dos runs do Kaggle, via API (servidor) ----
api = wandb.Api()
for rid in ["ytvelu85", "zwizpaya"]:
    print(f"\n===== KAGGLE {rid} =====")
    try:
        r = api.run(f"{ENT}/{PROJ}/{rid}")
        print("summary:", {k: r.summary.get(k) for k in ("teste", "l1", "_step", "_runtime")})
        print("history() rows     :", len(r.history(pandas=False)))
        print("scan_history() rows:", len(list(r.scan_history())))
    except Exception as e:
        print("err:", repr(e))

# ---- (B) run online LOCAL, do zero ----
print("\n===== LOCAL fresh run =====")
run = wandb.init(project=PROJ, entity=ENT, name="local-sanity")
rid, url = run.id, run.url
for i in range(10):
    run.log({"teste_local": i})
run.finish()
print("local run:", rid, url)

# linhas de filestream do log interno local (comparar com Kaggle)
print("\n--- filestream do log interno LOCAL ---")
try:
    log = sorted(glob.glob("wandb/**/debug-internal.log", recursive=True))[-1]
    for l in open(log, encoding="utf-8", errors="ignore").read().splitlines():
        if "filestream" in l:
            print(l)
except Exception as e:
    print("log err:", repr(e))

# ---- (C) ler de volta o run local via API (espera indexar) ----
time.sleep(20)
print("\n--- readback do run LOCAL via API (apos 20s) ---")
try:
    r = wandb.Api().run(f"{ENT}/{PROJ}/{rid}")
    print("summary:", {k: r.summary.get(k) for k in ("teste_local", "_step", "_runtime")})
    print("history() rows     :", len(r.history(pandas=False)))
    print("scan_history() rows:", len(list(r.scan_history())))
except Exception as e:
    print("err:", repr(e))
