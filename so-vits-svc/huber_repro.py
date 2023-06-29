from fairseq import checkpoint_utils
import intel_extension_for_pytorch as ipex
vec_path = "/ws1/zhaoqion/so-vits-svc/hubert/checkpoint_best_legacy_500.pt"
print("load model(s) from {}".format(vec_path))

models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
[vec_path],
suffix="",
)
model = models[0]
model.eval()

modelxpu = model.to("xpu")
print(modelxpu)
