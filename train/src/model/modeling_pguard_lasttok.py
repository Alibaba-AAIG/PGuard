# Token-level head with top-k layer selection, per-task classifiers, and ATC
# regularization. CE is computed at every nonzero marker-mask position.

from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertConfig, BertIntermediate, BertOutput

from src.utils.model_utils import BasicClassificationHead, NORMAL_LABEL, config_from_dict
from src.utils.modeling_outputs import PGuardMultiTasksOutputForCls


class AdapterBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, x):
        intermediate_output = self.intermediate(x)
        layer_output = self.output(intermediate_output, x)
        return layer_output


class PluginGuard(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.top_L = 3
        self.gate = nn.Linear(config.hidden_size, 1)
        self.gate_temperature = 1.0

        bert_layer_config = BertConfig()
        bert_layer_config.add_cross_attention = False

        self.input_proj = nn.Linear(config.hidden_size, bert_layer_config.hidden_size)
        self.adapter = nn.ModuleList( [AdapterBlock(bert_layer_config) for _ in range(3)] )  # FFN only, no attention needed

        self.norm = nn.LayerNorm(config.hidden_size, eps=bert_layer_config.layer_norm_eps)

        self.classifiers = nn.ModuleList()

        classification_heads = config.classifier["classification_heads"]
        for name, num_classes in classification_heads:
            # update config
            classifier_config = config_from_dict({
                "hidden_size": bert_layer_config.hidden_size,
                "dropout_prob": config.classifier["dropout_prob"],
                "num_classes":  num_classes,
                "dropout_type": config.classifier["dropout_type"]
            })
            classifier = BasicClassificationHead(classifier_config)
            self.classifiers.add_module(name, classifier)

        ### loss ###
        for head in config.task_schema:
            assert config.task_schema[head]['label2id'][NORMAL_LABEL] == 0, 'the id of the normal label must be 0: an implicit assumption in the loss below'

        self.lambda_intra_tv = 0.01
        self.lambda_intra_mono = 0.01
        self.lambda_intra = 1
        self.lambda_inter_tv = 0.01
        self.lambda_inter_mono = 0.01
        self.lambda_inter = 1

        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward_gate(self, features):
        # feature: (bs, layer, sent_len, hidden_size)

        B, L, S, H = features.shape

        scores = self.gate(features).squeeze(-1)   # (B, L, S)

        scores_no_noise = scores

        # Gumbel noise
        if self.training:
            U = torch.rand_like(scores)
            gumbel_noise = -torch.log(-torch.log(U + 1e-8) + 1e-8)
            scores += gumbel_noise

        # temperature controls exploration strength
        scores /= self.gate_temperature

        # topk selection
        topk_scores, topk_indices = torch.topk(scores, k=self.top_L, dim=1)  # (B, topk, S)

        # topk features
        expanded_indices = topk_indices.unsqueeze(-1).expand(B, self.top_L, S, H)
        topk_features = torch.gather(features, dim=1, index=expanded_indices)  #  (B, topk, S, H)

        weights = F.softmax(topk_scores, dim=1).unsqueeze(-1) # (B, topk, S, 1)
        final_feature = (topk_features * weights).sum(dim=1) # (B, S, H)

        all_scores = F.softmax(scores_no_noise, dim=1)

        return final_feature, topk_scores, topk_indices, all_scores

    def forward(
        self,
        features:torch.Tensor,  # (bs, layer, sent_len, hidden_size)
        token_type_ids:dict,    # dict(key=int, value=(bs, sent_len))
        label_type:str,
        label_marker:dict,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], PGuardMultiTasksOutputForCls]:
        '''token_type_ids represents the marker_mask; values >0 mark tokens that contribute to the loss'''

        features = self.norm(features)

        features, gate_topk_scores, gate_topk_indices, gate_all_scores = \
            self.forward_gate(features)  # (bs, sent_len, hidden_size), (B, topk, S), (B, topk, S), (B, L, S)

        features = self.input_proj(features)  # (bs, sent_len, inter_hidden_size)
        for layer in self.adapter:
            features = layer(features)  # (bs, sent_len, inter_hidden_size)

        full_logits_list = []
        for idx, classifier in enumerate(self.classifiers):
            logits = classifier(features)  # (bs, sent_len, num_classes)
            full_logits_list.append(logits)  # full logits

        attention_mask = None
        if token_type_ids is not None:
            device = features.device
            bs, sent_len = features.shape[:2]
            max_marker = len(token_type_ids)
            n_head = len(full_logits_list)
            marker_ids = list(token_type_ids.keys())
            assert all(int(marker_ids[i]) < int(marker_ids[i+1]) for i in range(len(marker_ids)-1)), 'marker ids must increase from left to right: they are used as a temporal ordering when computing the inter loss'

            token_type_ids = torch.stack([token_type_ids[k] for k in marker_ids], 0)  # (max_marker, bs, sent_len)

            # attention mask: positions of all markers
            attention_mask = (token_type_ids.sum(0) > 0).to(torch.long)  # (bs,sent_len)

        final_loss = None
        final_loss_dict = {'loss':None}
        if token_type_ids is not None and label_type is not None and label_marker is not None:
            if label_type != 'single':
                raise NotImplementedError(label_type)

            # stack markers into the bs dimension
            token_type_ids = token_type_ids.flatten(0,1)                              # (bs', sent_len)

            logits_list = [x.unsqueeze(0).expand(max_marker, -1, -1, -1) for x in full_logits_list]  # list of (max_marker, bs, sent_len, num_classes), len=n_head
            logits_list = [x.flatten(0,1) for x in logits_list]                                      # list of (bs', sent_len, num_classes), len=n_head  # expand returns a view, but the following flatten copies and costs memory; num_classes is at most ~100 here, so acceptable

            labels = [{id: item['batch_single_labels'][i] for id, item in label_marker.items()} for i in range(n_head)]  # list of dict(key=int, value=(bs)), len=n_head
            labels = [torch.stack([x[k] for k in marker_ids], 0) for x in labels]     # list of (max_marker, bs), len=n_head
            labels = [x.flatten(0,1) for x in labels]                        # list of (bs'), len=n_head
            labels = [x.unsqueeze(-1).expand(-1, sent_len) for x in labels]  # list of (bs', sent_len), len=n_head  # the label of each (bs, marker) pair must apply to the tokens with token_type_ids>0, hence the extra sent_len dim

            task_mask = {id: item['batch_task_mask'] for id, item in label_marker.items()}  # dict(key=int, value=(bs,n_head))
            task_mask = torch.stack([task_mask[k] for k in marker_ids], 0)    # (max_marker, bs, n_head)
            task_mask = task_mask.flatten(0,1)                                # (bs', n_head)

            if (token_type_ids > 0).sum() == 0:
                print(token_type_ids)
                print('No valid tokens remain in the marker fields of view; check truncation/max_length')
                final_loss_dict = {
                    'loss':torch.tensor(0.0, requires_grad=True, device=device),
                    'loss_ce':torch.tensor(0.0, requires_grad=True, device=device),
                    'loss_atc_intra':torch.tensor(0.0, requires_grad=True, device=device),
                    'loss_atc_inter':torch.tensor(0.0, requires_grad=True, device=device),
                    }
            else:
                # if some (bs, marker) pair has no valid token (truncated), its task_mask
                # must be zeroed; data preprocessing does not enforce this, so handle it here
                # sequences without any valid token get task_mask forced to 0
                task_mask *= ((token_type_ids > 0).sum(1) > 0).float().unsqueeze(-1)

                multitask_loss = []
                multitask_loss_ce = []
                multitask_loss_atc_intra = []
                multitask_loss_atc_inter = []
                for idx in range(n_head):
                    logits = logits_list[idx]   # (bs', sent_len, num_classes)  NOTE: bs' = max_marker * bs
                    labels_ = labels[idx]       # (bs', sent_len)

                    ### ce loss ###
                    loss_ce = nn.CrossEntropyLoss(reduction="none")(logits.flatten(0,1), labels_.flatten(0,1)).view(-1, sent_len)  # (bs',sent_len)

                    valid_token_mask = (token_type_ids > 0).float()  # (bs', sent_len)
                    loss_ce *= valid_token_mask  # (bs', sent_len)

                    valid_token_counts = valid_token_mask.sum(1).clamp(min=1e-8)  # (bs')
                    loss_ce = loss_ce.sum(1) / valid_token_counts  # (bs')

                    ### intra-marker tv & mono loss ###
                    # ATC penalizes variation and downward steps in harmful logits.
                    # Intra loss acts within each marker field of view; inter loss
                    # acts between adjacent markers' averaged harmful logits.
                    # implicit assumption: the normal label always has id 0, so take the max logits over labels starting from 1
                    harmful_logits, _ = torch.max(logits[..., 1:], dim=-1)  # (bs', sent_len, num_classes) --> (bs', sent_len, num_classes-1) --> (bs' sent_len)

                    continuous_mask_intra = valid_token_mask[:, :-1] * valid_token_mask[:, 1:]  # (bs', sent_len-1)

                    y_prev = harmful_logits[:, :-1]
                    y_curr = harmful_logits[:, 1:]

                    diffs = torch.abs(y_curr - y_prev)  # (bs', sent_len-1)
                    l_tv_intra = (diffs * continuous_mask_intra).sum(1) / continuous_mask_intra.sum(1).clamp(min=1e-8)

                    downward_steps = F.relu(y_prev - y_curr)  # (bs', sent_len-1)
                    l_mono_intra = (downward_steps * continuous_mask_intra).sum(1) / continuous_mask_intra.sum(1).clamp(min=1e-8)

                    loss_atc_intra = self.lambda_intra_tv * l_tv_intra + self.lambda_intra_mono * l_mono_intra  # (bs')

                    ### inter-marker tv & mono loss ###
                    masked_logits = harmful_logits * valid_token_mask  # (bs', sent_len)
                    marker_level_logits = masked_logits.sum(dim=-1) / valid_token_counts  # (bs')  each marker averages the tokens in its field of view

                    marker_level_logits = marker_level_logits.view(max_marker, bs).transpose(1, 0)  # (bs, max_marker)  swap dims: max_marker now plays the role of sent_len
                    marker_valid_mask = (valid_token_counts > 0).view(max_marker, bs).transpose(1, 0)  # (bs, max_marker)

                    continuous_mask_inter = marker_valid_mask[:, :-1] * marker_valid_mask[:, 1:]  # (bs, max_marker-1)

                    y_prev = marker_level_logits[:, :-1]
                    y_curr = marker_level_logits[:, 1:]

                    diffs = torch.abs(y_curr - y_prev)  # (bs, max_marker-1)
                    l_tv_inter = (diffs * continuous_mask_inter).sum(1) / continuous_mask_inter.sum(1).clamp(min=1e-8)

                    downward_steps = F.relu(y_prev - y_curr)  # (bs, max_marker-1)
                    l_mono_inter = (downward_steps * continuous_mask_inter).sum(1) / continuous_mask_inter.sum(1).clamp(min=1e-8)

                    loss_atc_inter = self.lambda_inter_tv * l_tv_inter + self.lambda_inter_mono * l_mono_inter  # (bs)
                    loss_atc_inter = loss_atc_inter.unsqueeze(0).expand(max_marker, -1).flatten()  # (bs')

                    loss = loss_ce + loss_atc_intra * self.lambda_intra + loss_atc_inter * self.lambda_inter
                    multitask_loss.append(loss)
                    multitask_loss_ce.append(loss_ce)
                    multitask_loss_atc_intra.append(loss_atc_intra)
                    multitask_loss_atc_inter.append(loss_atc_inter)

                final_loss_dict = {'loss':multitask_loss, 'loss_ce':multitask_loss_ce, 'loss_atc_intra':multitask_loss_atc_intra, 'loss_atc_inter':multitask_loss_atc_inter}
                for k, mt_loss in final_loss_dict.items():
                    mt_loss = torch.stack(mt_loss, 1) # (bs', n_head)
                    mt_loss *= task_mask

                    valid_task_counts = task_mask.float().sum().clamp(min=1e-8)
                    final_loss = mt_loss.sum() / valid_task_counts  # the final loss must not average over the bs dimension directly; task_mask counts the truly valid samples

                    final_loss_dict[k] = final_loss


            # restore shapes
            token_type_ids = token_type_ids.view(max_marker, bs, sent_len)


        return PGuardMultiTasksOutputForCls(
            classification_logits=None,
            full_classification_logits=full_logits_list,  # list of (bs, sent_len, num_classes), len=n_head
            attention_mask=attention_mask,  # (bs, sent_len)
            hidden_states=features,
            gate_topk_scores=gate_topk_scores,  # (B, topk, S)
            gate_topk_indices=gate_topk_indices,  # (B, topk, S)
            gate_all_scores=gate_all_scores,  # (B, L, S)
            loss=final_loss_dict['loss'],
            loss_dict=final_loss_dict,  # for visualization
            input_ids=kwargs.get('input_ids'),  # (bs, sent_len)
            token_type_ids=token_type_ids,  # (max_marker, bs, sent_len)
        )
