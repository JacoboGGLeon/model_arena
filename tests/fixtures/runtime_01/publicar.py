from pathlib import Path
import argparse
from observatorio_athena.common import Config
from observatorio_athena.prepare import prepare
from observatorio_athena.aws import Publisher


def main():
 parser=argparse.ArgumentParser(description='Preparar contratos MAC2/MAC3 y publicar una corrida para Athena')
 parser.add_argument('--run',required=True);parser.add_argument('--out',required=True)
 parser.add_argument('--config',default=str(Path(__file__).parent/'config/ada.json'))
 parser.add_argument('--delivery-dir',action='append',default=[])
 parser.add_argument('--training-history');parser.add_argument('--prepare-only',action='store_true')
 args=parser.parse_args();cfg=Config.read(args.config)
 prepare(args.run,args.out,cfg,delivery_dirs=args.delivery_dir,training_path=args.training_history)
 if not args.prepare_only:Publisher(cfg).publish(args.out)
if __name__=='__main__':main()
