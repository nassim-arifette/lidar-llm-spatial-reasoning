import torch
import torch.nn as nn
import math
from typing import List, Optional, Union
import torch.nn.functional as F
from models.base_model import BaseModel
from torch_geometric.nn.models import LightGCN
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import get_peft_model, LoraConfig




class Lidar_LLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lidar_encoder = Lidar_Encoder()
        self.vat = VAT(embed_dim=100, num_heads=5, num_views=6)
        self.mlp = MLP(100, 100)
        self.mlp.decoder = nn.Linear(100, 768)
        self.query = torch.rand((1, 100)).to(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.mps.is_available()
            else "cpu"
        )
        self.llm = LLM()

    def forward(self, lidar: torch.tensor, prompts: list[str],
                answer: list[str] = None, criterion=None,
                use_encoder=False,  view_idx: Optional[int] = None):
        self = self.to(lidar.device)
        self.query = self.query[0, :].repeat(lidar.size()[0], 1).to(lidar.device)
        if use_encoder:
            bev_feature = self.lidar_encoder(lidar)
        else:
            bev_feature = lidar
        bev_feature = self.vat(bev_feature, self.query, view_idx=view_idx)        
        bev_feature = self.mlp(bev_feature)
        output = self.llm(bev_feature, prompts)

        if answer is not None and criterion is not None:
            answer_tokens = self.llm.tokenizer(
                answer, padding=True, truncation=True, return_tensors="pt"
            ).input_ids

            loss = criterion(output[:, -1, :].view(-1, output.size(-1)),
                             answer_tokens.view(-1).to(lidar.device))

            valid_tokens = (answer_tokens != self.llm.tokenizer.pad_token_id).sum()

            return loss, valid_tokens
        return output

    def generate(self, lidar: torch.tensor, prompts: list[str], 
                 use_encoder=False, view_idx: Optional[int] = None):
        lidar = lidar.to(self.query.device)
        self.query = self.query[0, :].repeat(lidar.size()[0], 1).to(lidar.device)
        if use_encoder:
            bev_feature = self.lidar_encoder(lidar.to(self.query.device))
        else:
            bev_feature = lidar
        bev_feature = self.vat(bev_feature, self.query, view_idx=view_idx)
        bev_feature = self.mlp(bev_feature)
        gen = self.llm.generate(bev_feature, prompts)
        return gen


class MiniPointNet(nn.Module):
    def __init__(self, emb_size: int = 100):
        super(MiniPointNet, self).__init__()
        self.emb_size = emb_size

        self.mlp1 = nn.Linear(4, 64)
        self.mlp2 = nn.Linear(64, 128)
        self.mlp3 = nn.Linear(128, emb_size)

    def forward(self, x: torch.tensor):
        x = F.relu(self.mlp1(x))
        x = F.relu(self.mlp2(x))
        x = self.mlp3(x)
        x = torch.max(x, dim=0)[0]
        return x


class Lidar_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.pointnet = MiniPointNet()

    def forward(self, x: torch.tensor):
        x = self.pointnet(x)
        return x


class LLM(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["c_attn"],
            bias="none"
        )
        self.model = get_peft_model(self.model, self.lora_config)

        self.max_length = 100
        self.temperature = 0.7
        self.top_p = 0.9
        self.top_k = 50

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, x: torch.tensor, prompts: list[str]):
        inputs = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)

        input_embeds = self.model.get_input_embeddings()(inputs['input_ids'].to(x.device))

        batch_size = x.shape[0]
        seq_len = input_embeds.shape[1] + 1

        attention_mask = torch.zeros((batch_size, seq_len), dtype=torch.long, device=x.device)
        attention_mask[:, 0] = 1
        attention_mask[:, 1:1 + input_embeds.shape[1]] = inputs['attention_mask'].to(x.device)

        x = x.unsqueeze(1)
        input_embeds_with_prompt = torch.cat((x, input_embeds), dim=1)

        outputs = self.model(inputs_embeds=input_embeds_with_prompt, attention_mask=attention_mask)

        return outputs.logits

    def generate(self,
                 x: torch.Tensor,
                 prompts: Union[str, List[str]],
                 max_length: Optional[int] = None,
                 temperature: Optional[float] = None,
                 top_p: Optional[float] = None,
                 top_k: Optional[int] = None,
                 num_return_sequences: int = 1,
                 do_sample: bool = True,
                 batch_size: Optional[int] = None) -> List[str]:
        max_length = max_length or self.max_length
        temperature = temperature or self.temperature
        top_p = top_p or self.top_p
        top_k = top_k or self.top_k

        device = next(self.parameters()).device
        x = x.to(device)

        if isinstance(prompts, str):
            prompts = [prompts] * x.shape[0]

        if len(prompts) != x.shape[0]:
            raise ValueError(f"Le nombre de prompts ({len(prompts)}) doit correspondre "
                             f"à la taille du batch des nuages de points ({x.shape[0]})")

        if batch_size is not None and batch_size < x.shape[0]:
            all_generated_texts = []

            for i in range(0, x.shape[0], batch_size):
                batch_x = x[i:i+batch_size]
                batch_prompts = prompts[i:i+batch_size]

                batch_texts = self._generate_batch(
                    batch_x, batch_prompts, max_length, temperature,
                    top_p, top_k, num_return_sequences, do_sample
                )
                all_generated_texts.extend(batch_texts)

            return all_generated_texts
        else:
            return self._generate_batch(
                x, prompts, max_length, temperature,
                top_p, top_k, num_return_sequences, do_sample
            )

    def _generate_batch(self,
                        x: torch.Tensor,
                        prompts: List[str],
                        max_length: int,
                        temperature: float,
                        top_p: float,
                        top_k: int,
                        num_return_sequences: int,
                        do_sample: bool) -> List[str]:
        device = next(self.parameters()).device

        tokenized = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
        input_ids = tokenized['input_ids'].to(device).long()
        attention_mask = tokenized['attention_mask'].to(device)

        combined_embeds = self.model.get_input_embeddings()(input_ids)
        x = x.unsqueeze(1)
        combined_embeds = torch.cat([x, combined_embeds], dim=1)
        batch_size = input_ids.size(0)
        additional_mask = torch.ones(batch_size, 1).to(x.device)
        combined_attention_mask = torch.cat([additional_mask, attention_mask], dim=1)

        self.eval()
        with torch.no_grad():
            gen_kwargs = {
                "max_length": max_length,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "num_return_sequences": num_return_sequences,
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "attention_mask": combined_attention_mask
            }

            generated_sequences = self.model.generate(
                inputs_embeds=combined_embeds,
                **gen_kwargs
            )

            generated_texts = []
            for seq in generated_sequences:
                decoded_text = self.tokenizer.decode(seq, skip_special_tokens=True)
                generated_texts.append(decoded_text)
            return [s1 + " : " + s2 for s1, s2 in zip(prompts, generated_texts)]



class MLP(nn.Module):
    def __init__(self, inputdim: int, latentdim: int):
        super().__init__()
        self.encoder = nn.Linear(inputdim, latentdim)
        self.activation = nn.ReLU()
        self.decoder = nn.Linear(latentdim, inputdim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.tensor):
        x = self.encoder(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.decoder(x)


class CrossAttention(nn.Module):
    def __init__(self, embed_size: int, num_heads: int):
        super(CrossAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        assert self.head_dim * num_heads == embed_size, "Embedding size must be divisible by number of heads"

        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, bev_emb: torch.tensor,
                queries: torch.tensor, mask: torch.tensor = None):
        N = queries.shape[0]
        query = self.query(queries)
        keys = self.key(bev_emb)
        values = self.value(bev_emb)

        values = values.reshape(N, self.num_heads, self.head_dim).unsqueeze(1)
        keys = keys.reshape(N, self.num_heads, self.head_dim).unsqueeze(1)
        query = query.reshape(N, self.num_heads, self.head_dim).unsqueeze(1)
        energy = torch.einsum("blhd,bshd->blsh", [query, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy, dim=-1)

        out = torch.einsum("abcd,abce->abde", [attention, values])
        out = out.reshape(N, 1, self.num_heads * self.head_dim).squeeze(1)
        out = self.fc_out(out)
        return out


class LayerNorm(nn.Module):
    def __init__(self, embed_size: int, eps: float = 1e-6):
        super(LayerNorm, self).__init__()
        self.embed_size = embed_size
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(embed_size))
        self.beta = nn.Parameter(torch.zeros(embed_size))

    def forward(self, x: torch.tensor):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)

        x_norm = (x - mean) / torch.sqrt(variance + self.eps)

        out = self.gamma * x_norm + self.beta
        return out


class BatchNormWithPatching(nn.Module):
    def __init__(self, embed_size: int, eps: float = 1e-6,
                 momentum: float = 0.1):
        super(BatchNormWithPatching, self).__init__()
        self.embed_size = embed_size
        self.eps = eps
        self.momentum = momentum

        self.gamma = nn.Parameter(torch.ones(embed_size))
        self.beta = nn.Parameter(torch.zeros(embed_size))

        self.running_mean = torch.zeros(embed_size)
        self.running_var = torch.ones(embed_size)

    def forward(self, x: torch.tensor):
        if x.dim() == 2:
            batch_size = 1
            mean = x.mean(dim=0)
            variance = x.var(dim=0, unbiased=False)
        else:
            batch_size = x.size(0)
            mean = x.mean(dim=0, keepdim=True)
            variance = x.var(dim=0, keepdim=True, unbiased=False)

        if self.training:
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * variance
        else:
            mean = self.running_mean
            variance = self.running_var

        x_norm = (x - mean) / torch.sqrt(variance + self.eps)

        out = self.gamma * x_norm + self.beta
        return out


class VAT(nn.Module):
    def __init__(self, embed_dim=100, num_heads=5, num_views=6):
        super().__init__()
        self.crossAttention = CrossAttention(embed_dim, num_heads)
        self.selfAttention = CrossAttention(embed_dim, num_heads)
        self.mlp = MLP(embed_dim, embed_dim)
        self.num_views = num_views
        self.view_embeddings = nn.Parameter(torch.zeros(num_views, embed_dim))
    
    
    
    def forward(self, bev_emb, queries, view_idx: Optional[int] = None):
        
        """
        We have two choices here:
        A specific view mode:
        When view_idx is provided (if the sample is associated with for example the "front left" view),
        only the corresponding view embedding is added to the BEV feature before running cross-attention.
        
        A panoramic view mode:
        When no view index is provided (i.e a panoramic sample), 
        the code loops over all six view embeddings, injecting each one into a copy of the BEV feature, 
        processes each with cross-attention (with the same query embeddings), and finally averages the six resulting outputs. 
        This more closely matches the paper's 
        description of “injecting all six view position embeddings” rather than simply averaging the embeddings beforehand.
        
        
        """
        bev_emb = self.selfAttention(bev_emb, bev_emb)
        
        if view_idx is not None:
            view_embed = self.view_embeddings[view_idx]  
            view_embed = view_embed.unsqueeze(0).expand(bev_emb.size(0), -1)
            bev_emb = bev_emb + view_embed
            bev_emb = self.crossAttention(bev_emb, queries)
        else:
            outputs = []
            for i in range(self.num_views):
                view_embed = self.view_embeddings[i].unsqueeze(0).expand(bev_emb.size(0), -1)
                bev_emb_i = bev_emb + view_embed
                out_i = self.crossAttention(bev_emb_i, queries)
                outputs.append(out_i)
            bev_emb = torch.stack(outputs, dim=0).mean(dim=0)
        
        return self.mlp(bev_emb)
