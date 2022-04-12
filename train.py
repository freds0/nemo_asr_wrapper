import argparse
import copy
import nemo
import nemo.collections.asr as nemo_asr
import pytorch_lightning as pl
try:
    from ruamel.yaml import YAML
except ModuleNotFoundError:
    from ruamel_yaml import YAML


def execute_train(args):

    config_path = args.config_file

    yaml = YAML(typ='safe')
    with open(config_path) as f:
        params = yaml.load(f)


    from omegaconf import DictConfig
    params['model']['train_ds']['manifest_filepath'] = args.train_file
    params['model']['train_ds']['batch_size'] = args.batch_size
    params['model']['validation_ds']['manifest_filepath'] = args.test_file
    params['model']['validation_ds']['batch_size'] = 4

    # This line will download pre-trained QuartzNet15x5 model from NVIDIA's NGC cloud and instantiate it for you
    model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En")
    # Point to the new validation data for fine-tuning
    #model.setup_validation_data(val_data_config=params['model']['validation_ds'])

    # Point to the data we'll use for fine-tuning as the training set
    #model=.setup_training_data(train_data_config=params['model']['train_ds'])


    # Let's add "!" symbol there. Note that you can (and should!) change the vocabulary
    # entirely when fine-tuning using a different language.
    model.change_vocabulary(
        new_vocabulary=[
            ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 
            's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'á', 'à', 'â', 'ã', 'é', 'ê', 'í', 'î', 'ó', 'ô', 'õ', 'ú', 'û', 'ü', 'ç'
        ]
    )

    new_opt = copy.deepcopy(params['model']['optim'])
    new_opt['lr'] = args.learning_rate

    # Use the smaller learning rate we set before
    model.setup_optimization(optim_config=DictConfig(new_opt))

    # Point to the data we'll use for fine-tuning as the training set
    model.setup_training_data(train_data_config=params['model']['train_ds'])

    # Point to the new validation data for fine-tuning
    model.setup_validation_data(val_data_config=params['model']['validation_ds'])

    print(model.summarize())

    trainer = pl.Trainer(gpus=args.num_gpus, max_epochs=args.num_epochs)

    trainer.fit(model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='./')
    parser.add_argument('-c', '--config_file', default='configs/config.yaml', help='Yaml config filepath.')
    parser.add_argument('-e', '--num_epochs', type=int, default=30)
    parser.add_argument('-b', '--batch_size', type=int, default=40)
    parser.add_argument('-g', '--num_gpus', type=int, default=1)
    parser.add_argument('-l', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-t', '--train_file', type=str, default='./train_manifest.csv')
    parser.add_argument('-s', '--test_file', type=str, default='./test_manifest.csv')
    args = parser.parse_args()

    execute_train(args)

if __name__ == "__main__":
    main()

