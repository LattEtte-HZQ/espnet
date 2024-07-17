"""Transducer joint network implementation."""

import torch

from espnet2.asr_transducer.joint_network import JointNetwork


class JointNetworkMTP(torch.nn.Module):

    def __init__(
            self,
            vocab_size,
            encoder_output_size,
            decoder_output_size,
            pred_num,
            **activation_parameters
    ) -> None:
        super().__init__()

        self.pred_num = pred_num
        self.multi_token_predictor = torch.nn.ModuleList(
            [
                JointNetwork(
                    vocab_size,
                    encoder_output_size,
                    decoder_output_size,
                    **activation_parameters
                )
                for _ in range(pred_num)
            ]
        )

    def forward(
            self,
            enc_out: torch.Tensor,
            dec_out: torch.Tensor,
    ) -> torch.Tensor:

        logits = []
        for i in range(self.pred_num):
            logits.append(self.multi_token_predictor[i](enc_out, dec_out))

        return torch.stack(logits)
