import torch.utils
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from kan import KANLinear
from transformers import PreTrainedModel, GenerationConfig
import torch.nn.functional as F

gpt2_model = AutoModelForCausalLM.from_pretrained(
    "openai-community/gpt2", attn_implementation="eager"
).to("cuda")

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")


class PreProcess(torch.nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def cos_sims2eigen_vals(self, vec2vec_matricies: torch.Tensor):
        batch_size = vec2vec_matricies.shape[0]
        cube_num = vec2vec_matricies.shape[1]
        pop_size = vec2vec_matricies.shape[2]

        cos_similarities = torch.stack(
            [
                torch.cosine_similarity(vec, matrix[0], dim=0)
                for matricies in torch.unbind(vec2vec_matricies, dim=0)
                for matrix in torch.unbind(matricies, dim=0)
                for vec in torch.unbind(matrix, dim=0)
            ],
            dim=0,
        ).view(batch_size, cube_num,pop_size)
        symmetric_matricies = torch.zeros(
            size=(batch_size, cube_num,pop_size,pop_size), device="cuda"
        )
        for index_start in torch.arange(start=0, end=pop_size, device="cuda"):
            batch_indices, cube_indeces, row_indices, col_indices = torch.meshgrid(
                torch.arange(start=0, end=batch_size, device="cuda"),
                torch.arange(start=0, end=cube_num, device="cuda"),
                torch.arange(start=0, end=pop_size, device="cuda"),
                torch.arange(start=index_start, end=pop_size, device="cuda"),
                indexing="ij",
            )

            symmetric_matricies[batch_indices, cube_indeces, row_indices,col_indices] = (
                cos_similarities[batch_indices, index_start,col_indices]
            )
        eigen_vals = torch.stack(
            [
                torch.linalg.eigvalsh(matrix, UPLO="U")
                for matrix in torch.unbind(symmetric_matricies, dim=2)
            ],
            dim=0,
        ).view(batch_size,cube_num,pop_size).mean(dim=2)
        return eigen_vals.view(batch_size,-1)

    @torch.no_grad()
    def unit_vecs(self, vecs):
        norms = torch.stack(
            [torch.linalg.vector_norm(vec) for vec in torch.unbind(input=vecs, dim=1)],
            dim=0,
        )
        unit_vectors = torch.stack(
            [
                vec.div(norm)
                for vec, norm in zip(
                    torch.unbind(input=vecs, dim=1), torch.unbind(input=norms, dim=0)
                )
            ]
        ).view(*vecs.shape)
        return unit_vectors

    @torch.no_grad()
    def vec2vec_matricies(self, unit_vectors):
        batch_size = unit_vectors.shape[0]
        pop_size = unit_vectors.shape[1]
        dim_size = unit_vectors.shape[2]
        vec2vec_matrix = torch.stack(
            [
                (vec_1.add(vec_2)).div(2)
                for matrix in torch.unbind(input=unit_vectors, dim=0)
                for vec_1 in torch.unbind(input=matrix, dim=0)
                for vec_2 in torch.unbind(input=matrix, dim=0)
            ],
            dim=0,
        ).view(batch_size, pop_size,pop_size, dim_size)
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
        self.leaky_relu_linear = torch.nn.Linear(48, 48)
        self.sigmoid_linear = torch.nn.Linear(48, 48)
        self.silu_linear = torch.nn.Linear(48, 48)
        self.elu_linear = torch.nn.Linear(48, 48)
        self.gelu_linear = torch.nn.Linear(48, 48)
        self.parametric_relu_linear = torch.nn.Linear(48, 48)
        self.tanh_linear = torch.nn.Linear(48, 48)
        self.mish_linear = torch.nn.Linear(48, 48)
        self.kan = KANLinear(48, 48)
        self.kan_output = KANLinear(9*48, 9*48)
        self.linear_output = torch.nn.Linear(9*48, 6)
        # -----------------------------------------------
        self.layer_norm = torch.nn.LayerNorm(9*48)
        self.softmax = torch.nn.Softmax(dim=1)
        self.drop_out = torch.nn.Dropout1d(p=0.4)

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
        x_final = self.layer_norm(x_final)
        x_final = self.drop_out(x_final)
        x_final = self.linear_output(x_final)
        x_final = self.softmax(x_final)
        return x_final


class Model(PreTrainedModel):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(gpt2_model.config, *args, **kwargs)
        self.gpt2 = gpt2_model.to("cuda")  # 768x50257 output
        self.config = self.gpt2.config
        self.preprocess = PreProcess()
        self.mfp = MFP()
        self.preprocess.requires_grad_(False)
        self.gpt2.requires_grad_(False)
        self.gen_config = GenerationConfig(
            max_length=1024,
            min_length=1024,
            do_sample=True,
            top_k=1,
            top_p=0.9,
            output_logits=True,
            pad_token_id=self.config.eos_token_id,
        )
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self,labels, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        attention_mask = attention_mask.to("cuda")
        input_ids = input_ids.to("cuda")
        input_ids = input_ids.mul(attention_mask)
        batch_size = input_ids.shape[0]
        dim_size = input_ids.shape[1]
        output = torch.stack(
            [
                self.gpt2.generate(
                    input_id.view(1, -1), generation_config=self.gen_config
                )
                for input_id in torch.unbind(input_ids, dim=0)
                for _ in torch.arange(start=0, end=3)
            ],
            dim=0,
        ).view(batch_size, 3, 1024)
        preprocessed = self.preprocess(output.float())
        padding = torch.zeros(size=(batch_size,48),device="cuda",dtype=torch.float32)
        padding[:batch_size,:preprocessed.shape[1]] = preprocessed
        final = self.mfp(padding)
        y = F.one_hot(labels.long(),num_classes=6).float()
        losses = self.loss(final,y)
        return (losses,final)


if __name__ == "__main__":

    Model().to("cuda")
