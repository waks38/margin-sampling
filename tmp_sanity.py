import time, wandb
ENT, PROJ = "felwaks-universidade-federal-de-minas-gerais", "margin-sampling"
print("wandb version:", wandb.__version__)
run = wandb.init(project=PROJ, entity=ENT, name="ver-test")
rid, url = run.id, run.url
for i in range(10):
    run.log({"vtest": i})
run.finish()
print("run:", rid, url)
time.sleep(20)
r = wandb.Api().run(f"{ENT}/{PROJ}/{rid}")
print("history() rows     :", len(r.history(pandas=False)))
print("scan_history() rows:", len(list(r.scan_history())))
