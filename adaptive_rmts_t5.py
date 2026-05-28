import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

from models.customized_modeling_t5 import CustomizedT5ForConditionalGeneration


class AdaptiveRMTSForConditionalGeneration(CustomizedT5ForConditionalGeneration):
    """Adaptive RMTS with exactly two auxiliary losses.

    Training objective:
        total_loss = generation_loss
                   + aux_loss_weight * aux_regression_loss
                   + load_balance_weight * load_balance_loss

    The view consistency loss is intentionally removed.
    """

    def __init__(
        self,
        config,
        num_traits=11,
        prompt_vocab_size=9,
        aux_loss_weight=0.1,
        load_balance_weight=0.1,
        view_dropout=0.1,
        fixed_equal_gate=False,
        gate_weights=None,
    ):
        super().__init__(config)
        self.num_traits = num_traits
        self.prompt_vocab_size = prompt_vocab_size
        self.aux_loss_weight = aux_loss_weight
        self.load_balance_weight = load_balance_weight
        self.view_dropout = view_dropout
        self.fixed_equal_gate = fixed_equal_gate
        # gate_weights: (gpt_weight, llama_weight), e.g., (0.5, 0.5) or (0.6, 0.4)
        self.gate_weights = gate_weights if gate_weights is not None else (0.5, 0.5)

        hidden_size = config.d_model
        self.prompt_embedding = nn.Embedding(prompt_vocab_size, hidden_size)
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Tanh(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(hidden_size, 2),
        )
        self.score_backbone = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.Tanh(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.score_head = nn.Linear(hidden_size, num_traits)
        self.prompt_score_heads = nn.ModuleList(
            [nn.Linear(hidden_size, num_traits) for _ in range(prompt_vocab_size)]
        )

    def _masked_mean(self, hidden_states, attention_mask):
        if attention_mask is None:
            return hidden_states.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (hidden_states * mask).sum(dim=1) / denom

    def _encode_inputs(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

    def _build_view_feature(self, essay_pool, rationale_pool, prompt_embed):
        feature = torch.cat([essay_pool, rationale_pool, prompt_embed], dim=-1)
        return self.score_backbone(feature)

    def _masked_regression_loss(self, predictions, targets, mask):
        mask = mask.to(predictions.dtype)
        loss = F.smooth_l1_loss(predictions, targets, reduction="none")
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)

    def _apply_score_head(self, feature, prompt_ids):
        shared_scores = self.score_head(feature)
        prompt_scores = torch.zeros_like(shared_scores)
        unique_prompt_ids = prompt_ids.unique(sorted=True)
        for prompt_id in unique_prompt_ids.tolist():
            prompt_mask = prompt_ids == prompt_id
            prompt_scores[prompt_mask] = self.prompt_score_heads[prompt_id](feature[prompt_mask])
        return shared_scores + prompt_scores

    def _apply_view_dropout(self, gpt_hidden, llama_hidden, gpt_pool, llama_pool):
        if (not self.training) or self.view_dropout <= 0:
            return gpt_hidden, llama_hidden, gpt_pool, llama_pool, None

        batch_size = gpt_hidden.size(0)
        keep_mask = torch.rand(batch_size, 2, device=gpt_hidden.device) > self.view_dropout
        dropped_all = ~keep_mask.any(dim=1)
        if dropped_all.any():
            keep_mask[dropped_all, 0] = True

        gpt_keep = keep_mask[:, 0].view(batch_size, 1, 1).to(gpt_hidden.dtype)
        llama_keep = keep_mask[:, 1].view(batch_size, 1, 1).to(llama_hidden.dtype)

        gpt_hidden = gpt_hidden * gpt_keep
        llama_hidden = llama_hidden * llama_keep
        gpt_pool = gpt_pool * gpt_keep.squeeze(1)
        llama_pool = llama_pool * llama_keep.squeeze(1)
        return gpt_hidden, llama_hidden, gpt_pool, llama_pool, keep_mask

    def encode_multi_view(
        self,
        essay_input_ids,
        essay_attention_mask,
        gpt_input_ids,
        gpt_attention_mask,
        llama_input_ids,
        llama_attention_mask,
        prompt_ids,
    ):
        essay_hidden = self._encode_inputs(essay_input_ids, essay_attention_mask)
        gpt_hidden = self._encode_inputs(gpt_input_ids, gpt_attention_mask)
        llama_hidden = self._encode_inputs(llama_input_ids, llama_attention_mask)

        essay_pool = self._masked_mean(essay_hidden, essay_attention_mask)
        gpt_pool = self._masked_mean(gpt_hidden, gpt_attention_mask)
        llama_pool = self._masked_mean(llama_hidden, llama_attention_mask)

        prompt_ids = prompt_ids.clamp(min=0, max=self.prompt_vocab_size - 1)
        prompt_embed = self.prompt_embedding(prompt_ids)

        gpt_hidden, llama_hidden, gpt_pool, llama_pool, keep_mask = self._apply_view_dropout(
            gpt_hidden,
            llama_hidden,
            gpt_pool,
            llama_pool,
        )

        gate_feature = torch.cat([essay_pool, gpt_pool, llama_pool, prompt_embed], dim=-1)
        if self.fixed_equal_gate:
            # 使用固定的门控权重
            gate_scores = torch.tensor(
                self.gate_weights,
                dtype=gate_feature.dtype,
                device=gate_feature.device,
            ).unsqueeze(0).expand(gate_feature.size(0), -1)
            if keep_mask is not None:
                masked_scores = gate_scores * keep_mask.to(gate_scores.dtype)
                gate_scores = masked_scores / masked_scores.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        else:
            gate_logits = self.gate_network(gate_feature)
            if keep_mask is not None:
                gate_logits = gate_logits.masked_fill(~keep_mask, -1e4)
            gate_scores = torch.softmax(gate_logits, dim=-1)

        fused_rationale = (
            gate_scores[:, 0].view(-1, 1, 1) * gpt_hidden
            + gate_scores[:, 1].view(-1, 1, 1) * llama_hidden
        )
        fused_hidden = self.proj(
            torch.cat([essay_hidden, fused_rationale], dim=1).transpose(1, 2)
        ).transpose(1, 2)
        fused_attention_mask = torch.ones(
            fused_hidden.size()[:2],
            dtype=essay_attention_mask.dtype,
            device=essay_attention_mask.device,
        )

        fused_pool = self._masked_mean(fused_hidden, fused_attention_mask)
        gpt_feature = self._build_view_feature(essay_pool, gpt_pool, prompt_embed)
        llama_feature = self._build_view_feature(essay_pool, llama_pool, prompt_embed)
        fused_feature = self._build_view_feature(essay_pool, fused_pool, prompt_embed)

        return {
            "encoder_outputs": BaseModelOutput(last_hidden_state=fused_hidden),
            "attention_mask": fused_attention_mask,
            "gate_scores": gate_scores,
            "fused_scores": self._apply_score_head(fused_feature, prompt_ids),
            "gpt_scores": self._apply_score_head(gpt_feature, prompt_ids),
            "llama_scores": self._apply_score_head(llama_feature, prompt_ids),
        }

    def predict_trait_scores(
        self,
        essay_input_ids,
        essay_attention_mask,
        gpt_input_ids,
        gpt_attention_mask,
        llama_input_ids,
        llama_attention_mask,
        prompt_ids,
    ):
        outputs = self.encode_multi_view(
            essay_input_ids=essay_input_ids,
            essay_attention_mask=essay_attention_mask,
            gpt_input_ids=gpt_input_ids,
            gpt_attention_mask=gpt_attention_mask,
            llama_input_ids=llama_input_ids,
            llama_attention_mask=llama_attention_mask,
            prompt_ids=prompt_ids,
        )
        return outputs["fused_scores"], outputs["gate_scores"]

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        essay_input_ids=None,
        essay_attention_mask=None,
        gpt_input_ids=None,
        gpt_attention_mask=None,
        llama_input_ids=None,
        llama_attention_mask=None,
        prompt_ids=None,
        trait_labels=None,
        trait_mask=None,
        **kwargs,
    ):
        if essay_input_ids is None or gpt_input_ids is None or llama_input_ids is None or prompt_ids is None:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                head_mask=head_mask,
                decoder_head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        multi_view_outputs = self.encode_multi_view(
            essay_input_ids=essay_input_ids,
            essay_attention_mask=essay_attention_mask,
            gpt_input_ids=gpt_input_ids,
            gpt_attention_mask=gpt_attention_mask,
            llama_input_ids=llama_input_ids,
            llama_attention_mask=llama_attention_mask,
            prompt_ids=prompt_ids,
        )

        generation_outputs = super().forward(
            input_ids=None,
            attention_mask=multi_view_outputs["attention_mask"],
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=multi_view_outputs["encoder_outputs"],
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )

        total_loss = generation_outputs.loss
        if trait_labels is not None and trait_mask is not None:
            aux_loss = self._masked_regression_loss(
                multi_view_outputs["fused_scores"],
                trait_labels,
                trait_mask,
            )
            average_gate = multi_view_outputs["gate_scores"].mean(dim=0)
            uniform_gate = torch.full_like(average_gate, 0.5)
            load_balance_loss = F.kl_div(
                average_gate.clamp_min(1e-8).log(),
                uniform_gate,
                reduction="batchmean",
            )
            # Two-auxiliary-loss version:
            #   main loss: generation_outputs.loss
            #   auxiliary 1: aux_loss, directly supervises fused_scores with trait labels
            #   auxiliary 2: load_balance_loss, prevents the adaptive gate from collapsing
            # Consistency loss is intentionally removed.
            total_loss = (
                total_loss
                + self.aux_loss_weight * aux_loss
                + self.load_balance_weight * load_balance_loss
            )

        return Seq2SeqLMOutput(
            loss=total_loss,
            logits=generation_outputs.logits,
            past_key_values=generation_outputs.past_key_values,
            decoder_hidden_states=generation_outputs.decoder_hidden_states,
            decoder_attentions=generation_outputs.decoder_attentions,
            cross_attentions=generation_outputs.cross_attentions,
            encoder_last_hidden_state=multi_view_outputs["encoder_outputs"].last_hidden_state,
            encoder_hidden_states=generation_outputs.encoder_hidden_states,
            encoder_attentions=generation_outputs.encoder_attentions,
        )
