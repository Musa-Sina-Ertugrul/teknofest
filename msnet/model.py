import torch.utils
import torch
from kan import KANLinear,KAN
import torch.nn.functional as F
from transformers import AutoModel,AutoModelForSequenceClassification,AutoModelForCausalLM

torch.manual_seed(42)
torch.cuda.manual_seed(42)

class MFP(torch.nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.leaky_relu = torch.nn.LeakyReLU()
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.mish = torch.nn.Mish()
        self.silu = torch.nn.SiLU()
        self.selu = torch.nn.SELU()
        self.elu = torch.nn.ELU()
        self.gelu = torch.nn.GELU()
        self.softplus = torch.nn.Softplus()
        # -----------------------------------------------
        self.leaky_relu_linear = torch.nn.Linear(1280, 768)
        self.tanh_linear = torch.nn.Linear(1280,768)
        self.sigmoid_linear = torch.nn.Linear(1280, 768)
        self.softplus_linear = torch.nn.Linear(1280,768)
        self.mish_linear = torch.nn.Linear(1280, 768)
        self.selu_linear = torch.nn.Linear(1280,768)
        self.elu_linear = torch.nn.Linear(1280, 768)
        self.gelu_linear = torch.nn.Linear(1280,768)
        self.kan = KAN([2*768,768,2*768],spline_order=6,grid_size=10)
        # -----------------------------------------------
        self.linear_1 = torch.nn.Linear(8*768,8*512)
        self.linear_2 = torch.nn.Linear(8*512,2*768)
        self.linear_start_1 = torch.nn.Linear(768,10)
        self.linear_final_1 = torch.nn.Linear(2*768,1280)
        self.linear_final_2 = torch.nn.Linear(1280,3)
        # -----------------------------------------------
        self.layer_norm_start = torch.nn.LayerNorm(768,eps=1e-8)
        self.layer_norm_final = torch.nn.LayerNorm(8*768,eps=1e-8)
        self.drop_out = torch.nn.Dropout1d(p=0.1)
        # -----------------------------------------------
        self.softmax = torch.nn.Softmax(dim=1)
        self.flatten = torch.nn.Flatten()

    def forward(self, x):
        x = self.layer_norm_start(x)
        x = self.drop_out(x)
        x = self.flatten(self.linear_start_1(x))
        #-------------------------------------------
        x_leaky_relu = self.leaky_relu_linear(x)
        x_leaky_relu = self.leaky_relu(x_leaky_relu)
        # ------------------------------------------
        x_tanh = self.tanh_linear(x)
        x_tanh = self.tanh(x_tanh)
        # ------------------------------------------
        x_sigmoid = self.sigmoid_linear(x)
        x_sigmoid = self.sigmoid(x_sigmoid)
        # ------------------------------------------
        x_softplus = self.softplus_linear(x)
        x_softplus = self.softplus(x_softplus)
        #-------------------------------------------
        x_mish = self.mish_linear(x)
        x_mish = self.mish(x_mish)
        # ------------------------------------------
        x_selu = self.selu_linear(x)
        x_selu = self.selu(x_selu)
        # ------------------------------------------
        x_elu = self.elu_linear(x)
        x_elu = self.elu(x_elu)
        # ------------------------------------------
        x_gelu = self.gelu_linear(x)
        x_gelu = self.gelu(x_gelu)
        # ------------------------------------------
        x_final = torch.cat((x_mish,x_tanh,x_sigmoid,x_softplus,x_mish,x_selu,x_elu,x_gelu),dim=1)
        # ------------------------------------------
        x_final = self.layer_norm_final(x_final)
        x_final = self.drop_out(x_final)
        x_final = self.linear_1(x_final)
        x_final = self.elu(x_final)
        x_final = self.linear_2(x_final)
        x_final = self.elu(x_final)
        x_final = self.kan(x_final)
        x_final = self.linear_final_1(x_final)
        x_final = self.elu(x_final)
        x_final = self.linear_final_2(x_final)
        x_final = self.gelu(x_final)
        x_final = self.softmax(x_final)
        return x_final

class Model(torch.nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mfp = MFP()
        self.model = AutoModelForCausalLM.from_pretrained("dbmdz/bert-base-turkish-cased")
        self.model.requires_grad_(True)
        self.model.cls.predictions.decoder = torch.nn.Flatten(start_dim=2)
        self.model.cls.predictions.decoder.requires_grad_(True)

    def forward(self,input_ids: torch.Tensor):
        input_ids = input_ids.to("cpu").long()
        output = self.model(input_ids).logits
        output = self.mfp(output)
        return output

class DataSet(torch.utils.data.Dataset):
    def __init__(self,x,y):
        super(DataSet,self).__init__()
        self.x = x
        self.y = y
        
    def __getitem__(self,index):
        return self.x[index],self.y[index]
    
    def __len__(self):
        return len(self.x)

if __name__ == "__main__":

    Model().to("cuda")
