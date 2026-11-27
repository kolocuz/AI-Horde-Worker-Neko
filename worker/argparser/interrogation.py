"""Arg parsing for the main script."""
from worker import disable_xformers
from worker.argparser.framework import LazyArgs, arg_parser

arg_parser.add_argument(
    "-f",
    "--forms",
    nargs="+",
    type=str,
    required=False,
    default=None,
    help="Which forms to run on this horde.",
)

def _post_parse(_parsed):
    disable_xformers.activate()


args = LazyArgs(arg_parser, post_parse=_post_parse)
