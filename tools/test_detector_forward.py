import torch
from hubconf import dinov3_vit7b16_de


def main():
    model = dinov3_vit7b16_de(pretrained=False)
    model.to(torch.device('cpu'))
    model.eval()
    dummy = torch.randn(1,3,224,224)
    with torch.inference_mode():
        out = model(dummy)
    print('Detector forward OK; output type:', type(out))
    try:
        print('Output keys:', out.keys() if hasattr(out, 'keys') else type(out))
    except Exception:
        print(out)

if __name__ == '__main__':
    main()
