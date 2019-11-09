import click


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data feature engineering scripts to turn interim data from
        (../interim) into data which is ready for usage in ML models
        (saved in ../processed).
    """
    # <TODO>
    # ALL FEATURE ENGINEERING GOES IN HERE
    # OUTPUT COULD BE TWO SETS (ONE FOR BOOSTING, OTHER FOR NN)


if __name__ == '__main__':
    main()
