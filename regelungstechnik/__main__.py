from .cli import cli
from .gui import main as gui_main
import sys

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'gui':
        gui_main()
    else:
        cli()
