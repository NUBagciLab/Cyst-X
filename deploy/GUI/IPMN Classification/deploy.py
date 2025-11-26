import torch
from model import get_model
from monai.transforms import LoadImage, Resize, EnsureChannelFirst, Compose, ScaleIntensity

if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device")
    
    label = ['no risk', 'low risk', 'high risk']
    load_image = Compose([LoadImage(image_only=True), ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96))])

    model = get_model(out_channels = 3)
    model.load_state_dict(torch.load('./model_t1.pth', map_location='cpu', weights_only=True))
    model.to(device)
    model.eval()
    
    image = load_image('./AHN05.nii.gz').to(device)
    
    with torch.no_grad(): 
        output = model(image.unsqueeze(0))
        pred_label = label[torch.argmax(output)]
        print('The case is '+pred_label)
        pred_prob = torch.nn.functional.softmax(output, dim=1).cpu().numpy().squeeze()
        for i in range(3):
            print(f"Probability of {label[i]}: {pred_prob[i]*100:.2f}%")