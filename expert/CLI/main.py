import expert.CLI.CLI_utils
import sys
import os


def main():
	parser = expert.CLI.CLI_utils.get_CLI_parser()
	#parser.print_help()
	args = parser.parse_args()
	if args.mode == 'init':
		from expert.CLI.main_init import init
		init(args)
		sys.exit(0)
	elif args.mode == 'download':
		from expert.CLI.main_download import download
		download(args)
		sys.exit(0)
	elif args.mode == 'construct':
		from expert.CLI.main_construct import construct
		construct(args)
		sys.exit(0)
	elif args.mode == 'map':
		from expert.CLI.main_map import map
		map(args)
		sys.exit(0)
	elif args.mode == 'convert':
		from expert.CLI.main_convert import convert
		convert(args)
		sys.exit(0)
	elif args.mode == 'select':
		from expert.CLI.main_select import select
		select(args)
		sys.exit(0)
	elif args.mode == 'train':
		from expert.CLI.main_train import train
		train(args)
		sys.exit(0)
	elif args.mode == 'transfer':
		from expert.CLI.main_transfer import transfer
		transfer(args)
		sys.exit(0)
	elif args.mode == 'search':
		from expert.CLI.main_search import search
		search(args)
		sys.exit(0)
	else:
		raise RuntimeError('Please specify correct work mode, see `--help`.')

#main()
