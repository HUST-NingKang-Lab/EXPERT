from expert.CLI.CLI_utils import get_CLI_parser, get_CFG_reader
import sys
import os


def main():
	parser = get_CLI_parser()
	args = parser.parse_args()
	cfg = get_CFG_reader()
	if args.self_normalize:
		raise NotImplementedError()

	if args.mode == 'init':
		from expert.CLI.main_init import init
		init(cfg, args)
		sys.exit(0)
	elif args.mode == 'download':
		from expert.CLI.main_download import download
		download(cfg, args)
		sys.exit(0)
	elif args.mode == 'construct':
		from expert.CLI.main_construct import construct
		construct(cfg, args)
		sys.exit(0)
	elif args.mode == 'map':
		from expert.CLI.main_map import map
		map(cfg, args)
		sys.exit(0)
	elif args.mode == 'convert':
		from expert.CLI.main_convert import convert
		convert(cfg, args)
		sys.exit(0)
	elif args.mode == 'select':
		from expert.CLI.main_select import select
		select(cfg, args)
		sys.exit(0)
	elif args.mode == 'train':
		from expert.CLI.main_train import train
		train(cfg, args)
		sys.exit(0)
	elif args.mode == 'transfer':
		from expert.CLI.main_transfer import transfer
		transfer(cfg, args)
		sys.exit(0)
	elif args.mode == 'search':
		from expert.CLI.main_search import search
		search(cfg, args)
		sys.exit(0)
	elif args.mode == 'evaluate':
		from expert.CLI.main_evaluate import evaluate
		evaluate(cfg, args)
		sys.exit(0)
	else:
		raise RuntimeError('Please specify correct work mode, see `--help`.')

#main()
