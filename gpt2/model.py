import torch.utils
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from kan import KANLinear
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_utils import PushToHubMixin


gpt2_tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
gpt2_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")


class PreProcess(torch.nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def cos_sims2eigen_vals(self, vec2vec_matricies: torch.Tensor):
        batch_num = vec2vec_matricies.shape[0]
        row_num = vec2vec_matricies.shape[1]
        col_num = vec2vec_matricies.shape[2]
        cos_similarities = torch.stack(
            [
                torch.cosine_similarity(vec, matrix[0],dim=0)
                for matrix in torch.unbind(vec2vec_matricies, dim=0)
                for vec in torch.unbind(matrix, dim=0)
            ],
            dim=0,
        ).view(batch_num, row_num)
        symmetric_matricies = torch.zeros(
            size=(batch_num, row_num, col_num), device="cuda"
        )
        for index_start in torch.arange(start=0, end=col_num, device="cuda"):
            symmetric_matricies[
                torch.arange(start=0, end=batch_num, device="cuda"),
                torch.arange(start=0, end=row_num, device="cuda"),
                torch.arange(start=index_start, end=col_num, device="cuda"),
            ] = cos_similarities[
                torch.arange(start=0, end=batch_num, device="cuda"), index_start:
            ]
        eigen_vals = torch.stack(
            [
                torch.linalg.eigvalsh(matrix, UPLO="U")
                for matrix in torch.unbind(symmetric_matricies, dim=0)
            ],
            dim=0,
        )
        return eigen_vals

    @torch.no_grad()
    def unit_vecs(self, vecs):
        norms = torch.stack(
            [torch.linalg.vector_norm(vec) for vec in torch.unbind(input=vecs, dim=0)],
            dim=0,
        )
        unit_vectors = torch.stack(
            [
                vec.div(norm)
                for vec, norm in zip(
                    torch.unbind(input=vecs, dim=0), torch.unbind(input=norms, dim=0)
                )
            ]
        )
        return unit_vectors

    @torch.no_grad()
    def vec2vec_matricies(self, unit_vectors):
        batch_size = unit_vectors.shape[0]
        dim_size = unit_vectors.shape[1]
        vec2vec_matrix = torch.stack(
            [
                (vec_1 + vec_2).div(2)
                for vec_1 in torch.unbind(input=unit_vectors, dim=0)
                for vec_2 in torch.unbind(input=unit_vectors, dim=0)
            ],
            dim=0,
        ).view(batch_size, batch_size, dim_size)
        return vec2vec_matrix

    @torch.no_grad()
    def forward(self, input: torch.Tensor):
        unit_vecs = self.unit_vecs(input)
        vec2vec_matricies = self.vec2vec_matricies(unit_vecs)
        eigen_vals = self.cos_sims2eigen_vals(vec2vec_matricies)
        return eigen_vals


class MFP(torch.nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.leaky_relu = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.silu = torch.nn.SiLU()
        self.elu = torch.nn.ELU()
        self.gelu = torch.nn.GELU()
        self.parametric_relu = torch.nn.PReLU()
        self.tanh = torch.nn.Tanh()
        self.mish = torch.nn.Mish()
        # -----------------------------------------------
        self.leaky_relu_linear = torch.nn.Linear(768, 48)
        self.sigmoid_linear = torch.nn.Linear(768, 48)
        self.silu_linear = torch.nn.Linear(768, 48)
        self.elu_linear = torch.nn.Linear(768, 48)
        self.gelu_linear = torch.nn.Linear(768, 48)
        self.parametric_relu_linear = torch.nn.Linear(768, 48)
        self.tanh_linear = torch.nn.Linear(768, 48)
        self.mish_linear = torch.nn.Linear(768, 48)
        self.kan = KANLinear(768, 48)
        self.kan_output = KANLinear(384, 384)
        self.linear_output = torch.nn.Linear(384, 50257)
        # -----------------------------------------------
        self.batch_norm = torch.nn.BatchNorm1d(384)

    def forward(self, x):

        x_leaky_relu = self.leaky_relu_linear(x)
        x_leaky_relu = self.leaky_relu(x_leaky_relu)
        # ------------------------------------------
        x_sigmoid = self.sigmoid_linear(x)
        x_sigmoid = self.sigmoid(x_sigmoid)
        # ------------------------------------------
        x_silu = self.silu_linear(x)
        x_silu = self.silu(x_silu)
        # ------------------------------------------
        x_elu = self.elu_linear(x)
        x_elu = self.elu_linear(x_elu)
        # ------------------------------------------
        x_gelu = self.gelu_linear(x)
        x_gelu = self.gelu(x_gelu)
        # ------------------------------------------
        x_parametric_relu = self.parametric_relu_linear(x)
        x_parametric_relu = self.parametric_relu(x_parametric_relu)
        # ------------------------------------------
        x_tanh = self.tanh_linear(x)
        x_tanh = self.tanh(x_tanh)
        # ------------------------------------------
        x_mish = self.mish_linear(x)
        x_mish = self.mish_linear(x_mish)
        # ------------------------------------------
        x_kan = self.kan(x)
        # ------------------------------------------
        x_final = torch.cat(
            (
                x_leaky_relu,
                x_sigmoid,
                x_silu,
                x_elu,
                x_gelu,
                x_parametric_relu,
                x_tanh,
                x_mish,
                x_kan,
            ),
            dim=1,
        )
        x_final = self.kan_output(x_final)
        x_final = self.batch_norm(x_final)
        x_final = self.self.linear_output(x_final)
        return x_final


class Model(PreTrainedModel):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.gpt2 = gpt2_model  # 768x50257 output
        self.tokenizer = gpt2_tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.preprocess = PreProcess()
        self.mfp = MFP()
        self.gpt2.lm_head = torch.nn.Flatten()
        self.preprocess.requires_grad_(False)

    def forward(self, input: torch.Tensor):
        preprocessed = self.preprocess(input)
        output = self.gpt2(preprocessed)
        fine_tuning_section = self.mfp(output.logits)
        return fine_tuning_section


if __name__ == "__main__":

    Model().to("cuda")
