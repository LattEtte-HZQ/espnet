#!/usr/bin/env python3

from espnet2.tasks.asr_transducer_mtp import ASRTransducerMTPTask


def get_parser():
    """Get parser for ASR Transducer task."""
    parser = ASRTransducerMTPTask.get_parser()
    return parser


def main(cmd=None):
    r"""ASR Transducer training.

    Example:

        % python asr_transducer_train.py asr --print_config \
                --optim adadelta > conf/train_asr.yaml
        % python asr_transducer_train.py \
                --config conf/tuning/transducer/train_rnn_transducer.yaml
    """
    ASRTransducerMTPTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
