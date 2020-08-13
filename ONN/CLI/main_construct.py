from livingTree import SuperTree
from tqdm import tqdm


def construct(args):
	print('Reading microbiome structure...')
	with open(args.i, 'r') as f:
		con = f.read().splitlines()
	print('Generating Ontology...')
	otlg = SuperTree()
	str_cumsum = lambda x: [':'.join(x[0:i]) for i in range(1, len(x)+1)]
	paths = [str_cumsum(path.split(':'))[1:] for path in tqdm(con)]

	otlg.create_node(identifier='root')
	otlg.from_paths(paths=paths)
	if args.show:
		otlg.show()
	print('Done')
	otlg.to_pickle(args.o)