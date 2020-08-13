from ONN.src.utils import get_CLI_parser
import sys


def main():
	parser = get_CLI_parser()
	#parser.print_help()
	args = parser.parse_args()
	if args.mode == 'construct':
		from ONN.CLI.main_construct import construct
		construct(args)
		sys.exit(0)
	elif args.mode == 'map':
		from ONN.CLI.main_map import map
		map(args)
		sys.exit(0)
	elif args.mode == 'convert':
		from ONN.CLI.main_convert import convert
		convert(args)
		sys.exit(0)
	elif args.mode == 'select':
		from ONN.CLI.main_select import select
		select(args)
		sys.exit(0)
	elif args.mode == 'train':
		from ONN.CLI.main_train import train
		train(args)
		sys.exit(0)
	elif args.mode == 'transfer':
		from ONN.CLI.main_transfer import transfer
		transfer(args)
		sys.exit(0)
	elif args.mode == 'search':
		from ONN.CLI.main_search import search
		search(args)
		sys.exit(0)
	else:
		raise RuntimeError('Please specify correct work mode, see `--help`.')

#main()
