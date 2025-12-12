import inspect
import traceback
import torch
from types import ModuleType

import hubconf


def is_model(obj):
    try:
        import torch.nn as nn
        return isinstance(obj, nn.Module)
    except Exception:
        return False


def try_call_factory(name, fn):
    print(f"\n=== {name} ===")
    try:
        sig = inspect.signature(fn)
        kwargs = {}
        if "pretrained" in sig.parameters:
            kwargs["pretrained"] = False
        # try to call
        model = fn(**kwargs) if kwargs else fn()
        print("factory returned type:", type(model))
        if is_model(model):
            model.to(torch.device("cpu"))
            model.eval()
            # try a dummy forward if it accepts a tensor
            try:
                dummy = torch.randn(1, 3, 224, 224)
                # change: DetectorWithProcessor expects a list of (3,H,W) tensors
                if hasattr(model, "postprocessor") and hasattr(model, "detector"):
                    inp = [dummy.squeeze(0)]
                else:
                    inp = dummy
                with torch.inference_mode():
                    out = model(inp)
                print("forward OK; output type:", type(out))
                try:
                    print("output shape:", out.shape)
                except Exception:
                    pass
            except Exception as e:
                print("forward failed (may not accept image input):", e)
        else:
            print("not an nn.Module; skipping forward")
    except Exception:
        traceback.print_exc()


def main():
    # collect top-level callables from hubconf that are not the segmentor
    names = [n for n in dir(hubconf) if not n.startswith("_")]
    skip = {"dependencies"}
    for name in names:
        if name in skip:
            continue
        if "segment" in name.lower():
            print(f"Skipping segmentation-related: {name}")
            continue
        obj = getattr(hubconf, name)
        if callable(obj):
            try_call_factory(name, obj)


if __name__ == '__main__':
    main()
