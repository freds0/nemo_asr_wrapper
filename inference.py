import argparse
import os
import nemo
import nemo.collections.asr as nemo_asr
from glob import glob


def execute_inference(args):

    #model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name=args.model_path)
    if args.model_path.endswith('.ckpt'):
        model = nemo_asr.models.EncDecCTCModel.load_from_checkpoint(checkpoint_path=args.model_path)
    elif args.model_path.endswith('nemo'):
        model = nemo_asr.models.EncDecCTCModel.restore_from(restore_path=args.model_path)

    for filepath in sorted(glob(args.input_folder + '/*.wav')):
        transcription = model.transcribe(paths2audio_files=[filepath])[0]
        filename = os.path.basename(filepath)
        print(f"Transcriptfile {filename}: {transcription}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='./')
    parser.add_argument('-c', '--config_file', default='configs/config.yaml', help='Yaml config filepath.')
    parser.add_argument('-m', '--model_path', default='./lightning_logs/version_1/checkpoints/epoch=1-step=23203.ckpt', help='Yaml config filepath.')
    parser.add_argument('-i', '--input_folder', default='audios/', help='Audio folders')
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    args = parser.parse_args()

    execute_inference(args)


if __name__ == "__main__":
    main()
