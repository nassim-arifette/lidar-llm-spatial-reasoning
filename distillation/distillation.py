import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from models.model import LLM
from typing import List


class DistillationDataset(Dataset):
    def __init__(self,
                 lidar_embeddings: torch.tensor,
                 prompts: torch.tensor,
                 tokenizer,  # gpt tokenizer
                 max_length: int = 512):
        self.lidar_embeddings = lidar_embeddings
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        lidar_embedding = self.lidar_embeddings[idx]
        tokens = self.tokenizer(
            prompt, truncation=True, padding=True,
            max_length=self.max_length-1, return_tensors="pt"
        )

        return {
            "lidar_embedding": lidar_embedding,
            "input_ids": tokens.input_ids.squeeze(),
            "attention_mask": tokens.attention_mask.squeeze()
        }


class StudentGPT2(nn.Module):
    def __init__(self, teacher_config, num_layers_to_keep: int = 4):
        super().__init__()
        student_config = GPT2Config.from_dict(teacher_config.to_dict())
        student_config.n_layer = num_layers_to_keep

        self.student_model = GPT2LMHeadModel(student_config)

    def forward(self, input_ids, attention_mask, lidar_embedding=None,
                labels=None, output_hidden_states=True):
        if lidar_embedding is not None:
            embeddings = self.student_model.transformer.wte(input_ids)

            lidar_embedding = lidar_embedding.unsqueeze(1)
            combined_embeddings = torch.cat([lidar_embedding, embeddings], dim=1)

            extended_attention_mask = torch.ones(
                (attention_mask.shape[0], 1),
                device=attention_mask.device
            )
            extended_attention_mask = torch.cat([extended_attention_mask, attention_mask], dim=1)

            outputs = self.student_model.transformer(
                inputs_embeds=combined_embeddings,
                attention_mask=extended_attention_mask,
                output_hidden_states=output_hidden_states
            )

            hidden_states = outputs.last_hidden_state[:, 1:, :]

            lm_logits = self.student_model.lm_head(hidden_states)

            loss = None
            if labels is not None:
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            return {
                "loss": loss,
                "logits": lm_logits,
                "hidden_states": outputs.hidden_states if output_hidden_states else None
            }
        else:
            return self.student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=output_hidden_states
            )


def layer_distillation_loss(teacher_hidden_states: torch.tensor,
                            student_hidden_states: torch.tensor,
                            layer_mapping: List):
    loss = 0
    for student_idx, teacher_idx in layer_mapping.items():
        s_hidden = student_hidden_states[student_idx]
        t_hidden = teacher_hidden_states[teacher_idx]
        loss += F.mse_loss(s_hidden, t_hidden)
    return loss


def train_distilled_model(teacher_model, student_model, train_dataloader,
                          layer_mapping, num_epochs=3, lr=5e-5,
                          alpha=0.5, temperature=2.0,
                          device="cuda" if torch.cuda.is_available() else "mps"
                          if torch.mps.is_available() else "cpu"):

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=lr)
    teacher_model.to(device)
    student_model.to(device)
    teacher_model.eval()
    student_model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            lidar_embedding = batch["lidar_embedding"].to(device)

            with torch.no_grad():
                input_embeds = teacher_model.transformer.wte(input_ids)
                input_embeds_with_prompt = torch.cat((lidar_embedding.unsqueeze(1), input_embeds), dim=1)
                new_attention_mask = torch.cat(
                    (torch.ones((attention_mask.shape[0], 1), dtype=torch.long, device=device), attention_mask),
                    dim=1
                )
                teacher_outputs = teacher_model(
                    inputs_embeds=input_embeds_with_prompt,
                    attention_mask=new_attention_mask,
                    output_hidden_states=True
                )
                teacher_hidden_states = teacher_outputs.hidden_states

            student_outputs = student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                lidar_embedding=lidar_embedding,
                labels=input_ids,
                output_hidden_states=True
            )

            student_hidden_states = student_outputs["hidden_states"]
            task_loss = student_outputs["loss"]

            layer_loss = layer_distillation_loss(
                teacher_hidden_states,
                student_hidden_states,
                layer_mapping
            )

            loss = (1 - alpha) * task_loss + alpha * layer_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    return student_model


def main(outputfolder):
    num_student_layers = 4
    lidar_embedding_dim = 768
    batch_size = 16
    num_epochs = 3
    learning_rate = 5e-5
    alpha = 0.7
    temperature = 2.0
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

    teacher = LLM()
    # teacher.load_state_dict(...)
    tokenizer = teacher.tokenizer
    teacher_model = teacher.model

    student_model = StudentGPT2(
        teacher_config=teacher_model.config,
        num_layers_to_keep=num_student_layers
    )

    total_teacher_layers = teacher_model.config.n_layer
    layer_mapping = {}
    for i in range(num_student_layers):
        teacher_idx = int((i / (num_student_layers - 1)) * (total_teacher_layers - 1))
        layer_mapping[i] = teacher_idx

    lidar_embeddings = [torch.randn(lidar_embedding_dim) for _ in range(1000)]
    prompts = ["Tell me about this LiDAR data" for _ in range(1000)]

    dataset = DistillationDataset(lidar_embeddings, prompts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    trained_student = train_distilled_model(
        teacher_model=teacher_model,
        student_model=student_model,
        train_dataloader=dataloader,
        layer_mapping=layer_mapping,
        num_epochs=num_epochs,
        lr=learning_rate,
        alpha=alpha,
        temperature=temperature,
        device=device
    )

    torch.save(
        trained_student.state_dict(),
        outputfolder + "/distilled_lidar_gpt2.pt"
    )
