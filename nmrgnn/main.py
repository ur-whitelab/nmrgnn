import click


@click.command()
@click.option('--input', help='Input')
def main(input):
    """Simple program that greets NAME for a total of COUNT times."""
    raise NotImplementedError()


if __name__ == '__main__':
    main
