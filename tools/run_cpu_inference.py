import torch
import traceback

from dinov3.hub.backbones import dinov3_vits16


def main():
    try:
        device = torch.device("cpu")
        print("Device:", device)
        # instantiate a small backbone without pretrained weights to avoid downloads
        model = dinov3_vits16(pretrained=False)
        model.to(device)
        model.eval()

        dummy = torch.randn(1, 3, 224, 224, device=device)
        with torch.inference_mode():
            out = model(dummy)
        print("Output type:", type(out))
        try:
            print("Output shape:", out.shape)
            print("Output dtype:", out.dtype)
            print("Output stats: min=%s max=%s mean=%s" % (out.min().item(), out.max().item(), out.mean().item()))
        except Exception:
            print(out)
    except Exception:
        traceback.print_exc()


if __name__ == '__main__':
    main()
