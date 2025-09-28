import torch
from model import get_model
from monai.transforms import LoadImage, Resize, EnsureChannelFirst, Compose, ScaleIntensity
import argparse
import os 

if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Classification Training.")
    parser.add_argument('-i',"--input-dir", default='/home/pyq6817/Cyst-X/deploy/pipeline/example/test_preprocessed', required=True,type=str, help="images path")
    parser.add_argument('-m',"--modality", default="t1", required=True,type=str, help="modality")
    args = parser.parse_args()
    file_folder = args.input_dir
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device")
    
    label = ['no/low risk', 'high risk']
    load_image = Compose([LoadImage(image_only=True), ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96))])

    model = get_model(out_channels = 2)
    model_path = os.path.join(script_dir, f'model_{args.modality}.pth')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(device)
    model.eval()
    for file in os.listdir(file_folder):
        if file.endswith('nii.gz'):
            file_path = os.path.join(file_folder,file)
            image = load_image(file_path).to(device)

            with torch.no_grad(): 
                output = model(image.unsqueeze(0))
                pred_label = label[torch.argmax(output)]
                print('The case is '+pred_label)
                pred_prob = torch.nn.functional.softmax(output, dim=1).cpu().numpy().squeeze()
                for i in range(2):
                    print(f"Probability of {label[i]}: {pred_prob[i]*100:.2f}%")