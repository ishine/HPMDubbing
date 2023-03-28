import torch
import torch.nn as nn
from utils.tools import get_mask_from_lengths, generate_square_subsequent_mask

class HPM_DubbingLoss(nn.Module):
    """ HPM_Dubbing Loss """
    def __init__(self, preprocess_config, model_config):
        super(HPM_DubbingLoss, self).__init__()
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]

        self.loss_model = model_config["loss_function"]["model"]

        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.cross_loss = nn.CrossEntropyLoss()

    def forward(self, inputs, predictions):
        (
            mel_targets,
            _,
            max_mel_len,
            pitch_targets,
            energy_targets,
            duration_targets,
            _,
            emo_class_target,
            _,
            _,
            _,
            max_lip_lens,
            _,
        ) = inputs[6:]
        (
            mel_predictions,
            postnet_mel_predictions,
            pitch_predictions,
            energy_predictions,
            src_masks,
            mel_masks,
            _,
            _,
            attn_scores,
            emotion_prediction,
            max_src_len,
        ) = predictions

        src_masks = ~src_masks
        mel_masks = ~mel_masks
        log_duration_targets = torch.log(duration_targets.float() + 1)
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False

        if self.pitch_feature_level == "phoneme_level":
            pitch_predictions = pitch_predictions.masked_select(src_masks)
            pitch_targets = pitch_targets.masked_select(src_masks)
        elif self.pitch_feature_level == "frame_level":
            pitch_predictions = pitch_predictions.masked_select(mel_masks)
            pitch_targets = pitch_targets.masked_select(mel_masks)

        if self.energy_feature_level == "phoneme_level":
            energy_predictions = energy_predictions.masked_select(src_masks)
            energy_targets = energy_targets.masked_select(src_masks)
        if self.energy_feature_level == "frame_level":
            energy_predictions = energy_predictions.masked_select(mel_masks)
            energy_targets = energy_targets.masked_select(mel_masks)

        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        postnet_mel_predictions = postnet_mel_predictions.masked_select(
            mel_masks.unsqueeze(-1)
        )
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        mel_loss = self.mae_loss(mel_predictions, mel_targets)
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)

        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)

        if self.loss_model == 1:
            pitch_loss_mae = self.mae_loss(pitch_predictions, pitch_targets)
            energy_mae = self.mae_loss(energy_predictions, energy_targets)

            total_loss = (
                    mel_loss + postnet_mel_loss + pitch_loss + energy_loss + pitch_loss_mae + energy_mae
            )
            return (
                total_loss,
                mel_loss,
                postnet_mel_loss,
                pitch_loss,
                energy_loss,
                pitch_loss_mae,
                energy_mae,
            )

        elif self.loss_model == 2:
            mel_loss_mse = self.mse_loss(mel_predictions, mel_targets)
            postnet_mel_loss_mse = self.mse_loss(postnet_mel_predictions, mel_targets)

            total_loss = (
                    mel_loss + postnet_mel_loss + pitch_loss + energy_loss + mel_loss_mse + postnet_mel_loss_mse
            )
            return (
                total_loss,
                mel_loss,
                postnet_mel_loss,
                pitch_loss,
                energy_loss,
                mel_loss_mse,
                postnet_mel_loss_mse,
            )
        elif self.loss_model == 3:
            mel_loss_mse = self.mse_loss(mel_predictions, mel_targets)
            postnet_mel_loss_mse = self.mse_loss(postnet_mel_predictions, mel_targets)
            pitch_loss_mae = self.mae_loss(pitch_predictions, pitch_targets)
            energy_mae = self.mae_loss(energy_predictions, energy_targets)

            total_loss = (
                    mel_loss + postnet_mel_loss + pitch_loss + energy_loss +
                    pitch_loss_mae + energy_mae + mel_loss_mse + postnet_mel_loss_mse
            )
            return (
                total_loss,
                mel_loss,
                postnet_mel_loss,
                pitch_loss,
                energy_loss,
                pitch_loss_mae,
                energy_mae,
                mel_loss_mse,
                postnet_mel_loss_mse,
            )

        elif self.loss_model == 4:
            mel_loss_mse = self.mse_loss(mel_predictions, mel_targets)
            postnet_mel_loss_mse = self.mse_loss(postnet_mel_predictions, mel_targets)
            pitch_loss_mae = self.mae_loss(pitch_predictions, pitch_targets)
            energy_mae = self.mae_loss(energy_predictions, energy_targets)
            categories_loss = 0.3*self.cross_loss(emotion_prediction, emo_class_target)

            total_loss = (
                    mel_loss + postnet_mel_loss + pitch_loss + energy_loss +
                    pitch_loss_mae + energy_mae + mel_loss_mse + postnet_mel_loss_mse
                    + categories_loss
            )
            return (
                total_loss,
                mel_loss,
                postnet_mel_loss,
                pitch_loss,
                energy_loss,
                pitch_loss_mae,
                energy_mae,
                mel_loss_mse,
                postnet_mel_loss_mse,
                categories_loss,
            )
        else:
            total_loss = (
                    mel_loss + postnet_mel_loss + pitch_loss + energy_loss
            )
            return (
                total_loss,
                mel_loss,
                postnet_mel_loss,
                pitch_loss,
                energy_loss,
            )









