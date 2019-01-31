import click
from speechrecognition.config.config_reader import ConfigReader
from speechrecognition.main_predict import main_predict
from speechrecognition.main_train import main_train


@click.group()
@click.pass_context
def speech(ctx):
    if ctx.invoked_subcommand is None:
        click.echo('Missing speech subcommand! \n Choose train or predict command.')
    else:
        click.echo('Invoking command: %s' % ctx.invoked_subcommand)

@speech.command()
@click.option('-c', '--config', 'config_path', type=click.Path(exists=True), required=True, help='Configuration file for model.')
def train(config_path):
    config = ConfigReader(config_path)

    main_train(config)

@speech.command()
@click.option('-x', '--audio', type=click.Path(), required=True, help='Audio filename for speech prediction.')
@click.option('-c', '--config', 'config_path', type=click.Path(exists=True), required=True, help='Configuration file for model.')
def predict(audio, config_path):
    config = ConfigReader(config_path)

    print(audio)
    transcripted_text = main_predict(config, audio)
    print(transcripted_text)


if __name__ == '__main__':
    speech()



