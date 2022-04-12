import argparse
import json
import os
import librosa
from tqdm import tqdm

def build_manifest_file(input_file, output_file, min_duration):

    fin = open(input_file, 'r')
    fout = open(output_file, 'w') 

    for line_number, line in tqdm(enumerate(fin)):
        if line_number == 0:
            continue
        audio_path, transcript = line.split(',')
        transcript = transcript.strip()
        audio_path = audio_path.replace('/raid/fred/Wav2Vec-Wrapper/datasets/', '../Wav2Vec-Wrapper/datasets/')
        duration = librosa.core.get_duration(filename=audio_path)
        if duration < min_duration:
          continue
        # Write the metadata to the manifest
        metadata = {
            "audio_filepath": audio_path,
            "duration": duration,
            "text": transcript
        }
        #sjson.dump(metadata, fout, indent=4, ensure_ascii=False)
        json.dump(metadata, fout, ensure_ascii=False)
        fout.write('\n')

    fin.close()
    fout.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='./')
    parser.add_argument('--input_train', type=str, default='../Wav2Vec-Wrapper/datasets/metadata_train.csv')
    parser.add_argument('--input_test', type=str, default='../Wav2Vec-Wrapper/datasets/metadata_val.csv')
    parser.add_argument('--output_train', type=str, default='./train_manifest.csv')
    parser.add_argument('--output_test', type=str, default='./test_manifest.csv')
    parser.add_argument('--min_duration', type=float, default=2.0)
    args = parser.parse_args()

    # Building Manifests
    print("Building manifest files...")
    input_file = os.path.join(args.base_dir, args.input_train)
    output_file = os.path.join(args.base_dir,args.output_train)

    build_manifest_file(input_file, output_file, float(args.min_duration))
    print("Training manifest created.")

    input_file = os.path.join(args.base_dir, args.input_test)
    output_file = os.path.join(args.base_dir,args.output_test)

    build_manifest_file(args.input_test, args.output_test, float(args.min_duration))
    print("Test manifest created.")
    print("***Done***")


if __name__ == "__main__":
    main()
