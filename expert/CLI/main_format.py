from expert.src.proportions import Proportions


def format(cfg, args):
    prop = Proportions(read_from=args.input)
    krona_format = prop.krona_format()
    prop.export_krona(krona_format, output_dir=args.output)
