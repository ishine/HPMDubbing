import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

LRELU_SLOPE = 0.1

from transformer import Decoder, PostNet
from .modules import Affective_Prosody_Adaptor, Multi_head_Duration_Aligner, Scene_aTmos_Booster
from utils.tools import get_mask_from_lengths, generate_square_subsequent_mask
from style_models.Modules import Mish, LinearNorm, Conv1dGLU, MultiHeadAttention


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HPM_Dubbing(nn.Module):
    """ HPM_Dubbing """
    def __init__(self, preprocess_config, preprocess_config2, model_config):
        super(HPM_Dubbing, self).__init__()
        self.model_config = model_config
        self.loss_model = model_config["loss_function"]["model"]
        self.ln = nn.LayerNorm(256)
        self.style_encoder = MelStyleEncoder(model_config)
        self.MDA = Multi_head_Duration_Aligner(preprocess_config, model_config)
        self.APA = Affective_Prosody_Adaptor(preprocess_config, model_config)
        self.STB = Scene_aTmos_Booster(preprocess_config, model_config)

        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.pre_net_bottleneck = model_config["transformer"]["pre_net_bottleneck"]
        self.postnet = PostNet()
        self.n_speaker = 1
        if model_config["multi_speaker"]:
            with open(
                    os.path.join(
                        preprocess_config["path"]["preprocessed_path"], "speakers.json"
                    ),
                    "r",
            ) as f:
                self.n_speaker = len(json.load(f))
            with open(
                    os.path.join(
                        preprocess_config2["path"]["preprocessed_path"], "speakers.json"
                    ),
                    "r",
            ) as f:
                self.n_speaker += len(json.load(f))
            self.speaker_emb = nn.Embedding(
                self.n_speaker,
                model_config["transformer"]["encoder_hidden"],
            )
        self.n_emotion = 1
        self.Synchronization_coefficient = 4
        if model_config["with_emotion"]:
            self.n_emotion = preprocess_config["preprocessing"]["emotion"]["n_emotion"]
            self.emotion_emb = nn.Embedding(
                self.n_emotion + 1,
                model_config["transformer"]["encoder_hidden"],
                padding_idx=self.n_emotion,
            )
        self.dataset_name = preprocess_config["dataset"]

    def forward(
            self,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels=None,
            mel_lens=None,
            max_mel_len=None,
            p_targets=None,
            e_targets=None,
            d_targets=None,
            spks=None,
            emotions=None,
            emos=None,
            Feature_256=None,
            lip_lens = None,
            max_lip_lens = None,
            lip_embedding = None,
            p_control=1.0,
            e_control=1.0,
            d_control=1.0,
            useGT=None,
    ):
        """mask for voice, text, lip"""
        src_masks = get_mask_from_lengths(src_lens, max_src_len)  # tensor of True and False 16x249
        lip_masks = get_mask_from_lengths(lip_lens, max_lip_lens)
        if useGT:
            mel_masks = (
                get_mask_from_lengths(mel_lens, max_mel_len)
            )
        else:
            mel_masks = (
                get_mask_from_lengths(lip_lens*self.Synchronization_coefficient, max_lip_lens*self.Synchronization_coefficient)
            )
        """Extract Style Vector following V2C"""
        style_vector = self.style_encoder(mels, mel_masks)

        """Duration Aligner"""
        (output, attn_scores) = self.MDA(lip_embedding, lip_masks, texts, src_masks)

        """Add Style and emotion Vector following V2C"""
        if self.n_speaker > 1:
            if self.model_config["learn_speaker"]:
                output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
                    -1, max_src_len, -1)
            else:
                output = output + style_vector.unsqueeze(1).expand(-1, max_mel_len, -1)

        if self.dataset_name == "MovieAnimation":
            if self.n_emotion > 1:
                if self.model_config["learn_emotion"]:
                    output = output + self.emotion_emb(emotions).unsqueeze(1).expand(
                        -1, max_src_len, -1)
                else:
                    output = output + emos.unsqueeze(1).expand(-1, max_mel_len, -1)

        """Prosody Adaptor"""
        (output, p_predictions, e_predictions,) = self.APA(output, mel_masks, max_mel_len, p_targets, e_targets,
                                                        Feature_256, spks, p_control, e_control, d_control, useGT)


        """Atmosphere Booster"""
        if self.dataset_name == "MovieAnimation":
            if self.loss_model == 4:
                # Scene feature extrated by V2C
                E_scene = emos.unsqueeze(1).expand(-1, max_mel_len, -1)
                (output, emotion_prediction) = self.STB(output, E_scene, mel_masks, max_mel_len,)
            else:
                emotion_prediction = None
        else:
            emotion_prediction = None

        """Mel-Generator"""
        output, mel_masks = self.decoder(output, mel_masks)
        # output = output + style_vector.unsqueeze(1).expand(-1, max_mel_len, -1)  #
        output = self.mel_linear(output)
        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            src_masks,
            mel_masks,
            src_lens,
            lip_lens*self.Synchronization_coefficient,
            attn_scores,
            emotion_prediction,
            max_src_len,
        )


class MelStyleEncoder(nn.Module):
    ''' MelStyleEncoder '''

    def __init__(self, model_config):
        super(MelStyleEncoder, self).__init__()
        self.in_dim = model_config["Stylespeech"]["n_mel_channels"]
        self.hidden_dim = model_config["Stylespeech"]["style_hidden"]
        self.out_dim = model_config["Stylespeech"]["style_vector_dim"]
        self.kernel_size = model_config["Stylespeech"]["style_kernel_size"]
        self.n_head = model_config["Stylespeech"]["style_head"]
        self.dropout = model_config["Stylespeech"]["dropout"]

        self.spectral = nn.Sequential(
            LinearNorm(self.in_dim, self.hidden_dim),
            Mish(),
            nn.Dropout(self.dropout),
            LinearNorm(self.hidden_dim, self.hidden_dim),
            Mish(),
            nn.Dropout(self.dropout)
        )

        self.temporal = nn.Sequential(
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
        )

        self.slf_attn = MultiHeadAttention(self.n_head, self.hidden_dim,
                                           self.hidden_dim // self.n_head, self.hidden_dim // self.n_head, self.dropout)

        self.fc = LinearNorm(self.hidden_dim, self.out_dim)

    def temporal_avg_pool(self, x, mask=None):
        if mask is None:
            out = torch.mean(x, dim=1)
        else:
            len_ = (~mask).sum(dim=1).unsqueeze(1)
            x = x.masked_fill(mask.unsqueeze(-1), 0)
            x = x.sum(dim=1)
            out = torch.div(x, len_)
        return out

    def forward(self, x, mask=None):
        max_len = x.shape[1]
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1) if mask is not None else None

        # spectral
        x = self.spectral(x)
        # temporal
        x = x.transpose(1, 2)
        x = self.temporal(x)
        x = x.transpose(1, 2)
        # self-attention
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0)
        x, _ = self.slf_attn(x, mask=slf_attn_mask)
        # fc
        x = self.fc(x)
        # temoral average pooling
        w = self.temporal_avg_pool(x, mask=mask)

        return w




