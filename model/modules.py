import os
import json
import copy
import math
from collections import OrderedDict
from torch.nn.utils import weight_norm
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from utils.tools import get_mask_from_lengths, pad
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from utils.tools import init_weights, get_padding
from transformer import Encoder, Lip_Encoder
LRELU_SLOPE = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Multi_head_Duration_Aligner(nn.Module):
    """Multi_head_Duration_Aligner"""
    def __init__(self, preprocess_config, model_config):
        super(Multi_head_Duration_Aligner, self).__init__()
        self.dataset_name = preprocess_config["dataset"]
        self.encoder = Encoder(model_config)
        self.lip_encoder = Lip_Encoder(model_config)
        self.attn = nn.MultiheadAttention(256, 8, dropout=0.1)
        self.num_upsamples = len(model_config["upsample_ConvTranspose"]["upsample_rates"])
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(model_config["upsample_ConvTranspose"]["upsample_rates"],
                                       model_config["upsample_ConvTranspose"]["upsample_kernel_sizes"])):
            self.ups.append(weight_norm(
                ConvTranspose1d(model_config["upsample_ConvTranspose"]["upsample_initial_channel"],
                                model_config["upsample_ConvTranspose"]["upsample_initial_channel"], k,
                                u, padding=(u // 2 + u % 2), output_padding=u % 2)))
        resblock = ResBlock1 if model_config["upsample_ConvTranspose"]["resblock"] == '1' else ResBlock2
        self.num_kernels = len(model_config["upsample_ConvTranspose"]["resblock_kernel_sizes"])
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = model_config["upsample_ConvTranspose"]["upsample_initial_channel"]
            for j, (k, d) in enumerate(zip(model_config["upsample_ConvTranspose"]["resblock_kernel_sizes"],
                                           model_config["upsample_ConvTranspose"]["resblock_dilation_sizes"])):
                self.resblocks.append(resblock(ch, k, d))

    def forward(
            self,
            lip_embedding,
            lip_masks,
            texts,
            src_masks,
    ):
        output_lip = self.lip_encoder(lip_embedding, lip_masks)
        output_text = self.encoder(texts, src_masks)

        output, attn_scores = self.attn(query=output_lip.transpose(0, 1), key=output_text.transpose(0, 1),
                                        value=output_text.transpose(0, 1), key_padding_mask=src_masks)
        output = output.permute(1, 2, 0)
        # Chem_E2:
        # output = F.interpolate(output, scale_factor=4).to(device)

        # Chem_E4:
        for i in range(self.num_upsamples):
            output = F.leaky_relu(output, LRELU_SLOPE)
            output = self.ups[i](output)
            # xs = None
            # for j in range(self.num_kernels):
            #     if xs is None:
            #         xs = self.resblocks[i*self.num_kernels+j](output)
            #     else:
            #         xs += self.resblocks[i*self.num_kernels+j](output)
            # output = xs / self.num_kernels
        output = output.permute(0, 2, 1)


        return (output, attn_scores)


class Affective_Prosody_Adaptor(nn.Module):
    """Variance Adaptor"""

    def __init__(self, preprocess_config, model_config):
        super(Affective_Prosody_Adaptor, self).__init__()
        self.dataset_name = preprocess_config["dataset"]
        self.emo_fc_2_val = nn.Sequential(nn.Linear(256, 256),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(256, 256),
                                      nn.ReLU(inplace=True),
                                      )
        self.emo_fc_2_aro = nn.Sequential(nn.Linear(256, 256),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(256, 256),
                                      nn.ReLU(inplace=True),
                                      )

        self.W = nn.Linear(256, 256)
        self.Uo = nn.Linear(256, 256)
        self.Um = nn.Linear(256, 256)

        self.bo = nn.Parameter(torch.ones(256), requires_grad=True)
        self.bm = nn.Parameter(torch.ones(256), requires_grad=True)

        self.wo = nn.Linear(256, 1)
        self.wm = nn.Linear(256, 1)
        self.inf = 1e5

        self.loss_model = model_config["loss_function"]["model"]

        self.Valence = VariancePredictor_AV(model_config)
        self.Arousal = VariancePredictor_AV(model_config)

        self.pitch_predictor = VariancePredictor(model_config)
        self.energy_predictor = VariancePredictor(model_config)

        self.arousal_attention = VA_ScaledDotProductAttention(temperature=np.power(256, 0.5))

        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        self.scale_fusion = model_config["Affective_Prosody_Adaptor"]["Use_Scale_attention"]
        self.predictor_ = model_config["variance_predictor"]["predictor"]
        assert self.pitch_feature_level in ["phoneme_level", "frame_level"]
        assert self.energy_feature_level in ["phoneme_level", "frame_level"]

        pitch_quantization = model_config["variance_embedding"]["pitch_quantization"]
        energy_quantization = model_config["variance_embedding"]["energy_quantization"]
        n_bins = model_config["variance_embedding"]["n_bins"]
        assert pitch_quantization in ["linear", "log"]
        assert energy_quantization in ["linear", "log"]
        with open(
                os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            pitch_min, pitch_max = stats["pitch"][:2]
            energy_min, energy_max = stats["energy"][:2]

        if pitch_quantization == "log":
            self.pitch_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(pitch_min), np.log(pitch_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.pitch_bins = nn.Parameter(
                torch.linspace(pitch_min, pitch_max, n_bins - 1),
                requires_grad=False,
            )
        if energy_quantization == "log":
            self.energy_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(energy_min), np.log(energy_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.energy_bins = nn.Parameter(
                torch.linspace(energy_min, energy_max, n_bins - 1),
                requires_grad=False,
            )

        self.pitch_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["encoder_hidden"]
        )
        self.energy_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["encoder_hidden"]
        )

    def get_pitch_embedding(self, x, target, mask, control, useGT):
        prediction = self.pitch_predictor(x, mask)  # prediction for each src frame
        if useGT:
            embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
        else:
            prediction = prediction * control
            embedding = self.pitch_embedding(
                torch.bucketize(prediction, self.pitch_bins)
            )
        return prediction, embedding

    def get_energy_embedding(self, x, target, mask, control, useGT):
        prediction = self.energy_predictor(x, mask)
        if useGT:
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            prediction = prediction * control
            embedding = self.energy_embedding(
                torch.bucketize(prediction, self.energy_bins)
            )

        return prediction, embedding

    def forward(
            self,
            x,
            mel_mask=None,
            max_len=None,
            pitch_target=None,
            energy_target=None,
            Feature_256=None,
            spks=None,
            p_control=1.0,
            e_control=1.0,
            d_control=1.0,
            useGT=None,
    ):
        M = x
        valence = self.emo_fc_2_val(Feature_256)
        if self.scale_fusion:
            context_valence, _ = self.arousal_attention(M, valence, valence)  # torch.Size([32, 448, 256])
        else:
            sample_numb = valence.shape[1]
            W_f2d = self.W(M)
            U_objs = self.Uo(valence)
            attn_feat_V = W_f2d.unsqueeze(2) + U_objs.unsqueeze(
                1) + self.bo  # (bsz, sample_numb, max_objects, hidden_dim)
            attn_weights_V = self.wo(torch.tanh(attn_feat_V))  # (bsz, sample_numb, max_objects, 1)
            objects_mask_V = mel_mask[:, None, :, None].repeat(1, sample_numb, 1, 1).permute(0,2,1,3)  # (bsz, sample, max_objects_per_video, 1)
            attn_weights_V = attn_weights_V - objects_mask_V.float() * self.inf
            attn_weights_V = attn_weights_V.softmax(dim=-2)  # (bsz, sample_numb, max_objects, 1)
            attn_objects_V = attn_weights_V * attn_feat_V
            context_valence = attn_objects_V.sum(dim=-2)  # (bsz, sample_numb, hidden_dim)
        pitch_prediction, pitch_embedding = self.get_pitch_embedding(
            context_valence, pitch_target, mel_mask, p_control, useGT
        )
        x = x + pitch_embedding
        x = x + spks.unsqueeze(1).expand(-1, max_len, -1)

        Arousal = self.emo_fc_2_aro(Feature_256)
        if self.scale_fusion:
            context_arousal, _ = self.arousal_attention(M, Arousal, Arousal)
        else:
            sample_numb = Arousal.shape[1]
            W_f2d = self.W(M)
            U_motion = self.Um(Arousal)
            attn_feat = W_f2d.unsqueeze(2) + U_motion.unsqueeze(
                1) + self.bm  # (bsz, sample_numb, max_objects, hidden_dim)
            attn_weights = self.wm(torch.tanh(attn_feat))  # (bsz, sample_numb, max_objects, 1)
            objects_mask = mel_mask[:, None, :, None].repeat(1, sample_numb, 1, 1).permute(0, 2, 1,3)  # (bsz, sample, max_objects_per_video, 1)
            attn_weights = attn_weights - objects_mask.float() * self.inf
            attn_weights = attn_weights.softmax(dim=-2)  # (bsz, sample_numb, max_objects, 1)
            attn_objects = attn_weights * attn_feat
            context_arousal = attn_objects.sum(dim=-2)  # (bsz, sample_numb, hidden_dim)
        energy_prediction, energy_embedding = self.get_energy_embedding(
            context_arousal, energy_target, mel_mask, e_control, useGT
        )
        x = x + energy_embedding
        x = x + spks.unsqueeze(1).expand(-1, max_len, -1)

        return (
            x,
            pitch_prediction,
            energy_prediction,
        )


class Scene_aTmos_Booster(nn.Module):
    """Multi_head_Duration_Aligner"""
    def __init__(self, preprocess_config, model_config):
        super(Scene_aTmos_Booster, self).__init__()
        self.dataset_name = preprocess_config["dataset"]
        self.emo_fc_2_sence = nn.Sequential(nn.Linear(256, 128),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(128, 256),
                                            nn.ReLU(inplace=True),
                                            )
        self.Emo_attention = Emo_cross_attention(temperature=np.power(256, 0.5))
        self.Emo_predictor = Emo_VariancePredictor(model_config)

    def get_Emo_embedding(self, x, mask):
        emo_id, embedding = self.Emo_predictor(x, mask)  # prediction for each src frame
        return emo_id, embedding

    def forward(
            self,
            output,
            emos,
            mel_mask,
            max_len,
    ):
        emos = self.emo_fc_2_sence(emos)
        emo_context, _ = self.Emo_attention(emos, output, output)
        emotion_prediction, emo_embedding = self.get_Emo_embedding(
            emo_context, mel_mask)
        output = output + emo_embedding.unsqueeze(1).expand(-1, max_len, -1)


        return (output, emotion_prediction)

class LengthRegulator(nn.Module):
    """Length Regulator"""

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(device)

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len


class VariancePredictor_AV(nn.Module):
    """Duration, Pitch and Energy Predictor"""

    def __init__(self, model_config):
        super(VariancePredictor_AV, self).__init__()

        self.input_size = model_config["transformer"]["encoder_hidden"]
        self.filter_size = model_config["variance_predictor"]["filter_size"]
        self.kernel = model_config["variance_predictor"]["kernel_size"]
        self.conv_output_size = model_config["variance_predictor"]["filter_size"]
        self.dropout = model_config["variance_predictor"]["dropout"]

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=1,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

    def forward(self, encoder_output):
        out = self.conv_layer(encoder_output)
        return out


class Emo_cross_attention(nn.Module):
    """ Emo_cross_attention """
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        output = torch.bmm(attn, v)

        return output, attn


class VA_ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        output = torch.bmm(attn, v)

        return output, attn

class VariancePredictor(nn.Module):
    """Duration, Pitch and Energy Predictor"""

    def __init__(self, model_config):
        super(VariancePredictor, self).__init__()

        self.input_size = model_config["transformer"]["encoder_hidden"]
        self.filter_size = model_config["variance_predictor"]["filter_size"]
        self.kernel = model_config["variance_predictor"]["kernel_size"]
        self.conv_output_size = model_config["variance_predictor"]["filter_size"]
        self.dropout = model_config["variance_predictor"]["dropout"]

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=1,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
            w_init="linear",
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x


class Emo_VariancePredictor(nn.Module):
    """Duration, Pitch and Energy Predictor"""

    def __init__(self, model_config):
        super(Emo_VariancePredictor, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        # self.dropout_a = nn.Dropout(0.01)

        self.input_size = model_config["transformer"]["encoder_hidden"]
        self.filter_size = model_config["variance_predictor"]["filter_size"]
        self.kernel = model_config["variance_predictor"]["kernel_size"]
        self.conv_output_size = model_config["variance_predictor"]["filter_size"]
        self.dropout = model_config["variance_predictor"]["dropout"]

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=1,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)
        self.max_pool = nn.AdaptiveAvgPool1d(8)
        self.FC1 = nn.Linear(256, 128)
        self.FC2 = nn.Linear(128, 8)

    def forward(self, mel, mask):
        """
        mask is mel-mask
        Attention: Q:mel (mel_length, E-z) K:Scene (1, E-z)
        """
        out = self.conv_layer(mel)
        out_embedding = torch.mean(out, dim=1, keepdim=True)  # ([32, 1, 256])
        out_class = self.max_pool(out_embedding)
        out_class = out_class.squeeze(1)

        return out_class, out_embedding.squeeze(1)



class ResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x


class ResBlock2(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x
