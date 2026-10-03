"""Replay the final extension and path experiments without editing old proofs."""
import argparse
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import audit_daily_opportunity_typed as typed_auditor
import replay_daily_opportunity_accounts as base
from research_daily_graph_data import publish_manifest, sha256
from research_daily_opportunity_leadership import leadership_variant


def replay(source, out):
    source = Path(source)
    with ExitStack() as stack:
        if (source/'extension_manifest.json').exists():
            from research_daily_opportunity_extension_filter import extension_variant, validate_extension_inputs
            validate_extension_inputs(source)
            stack.enter_context(extension_variant())
        if (source/'path_filter_manifest.json').exists():
            from research_daily_opportunity_path_runner import validate_path_filter
            validate_path_filter(source)
        stack.enter_context(leadership_variant())
        stack.enter_context(patch.object(base, 'load_auditor', return_value=typed_auditor))
        result = base.replay_one(source, Path(out))
    result['wrapper_sha256'] = sha256(__file__)
    result['typed_auditor_sha256'] = sha256(typed_auditor.__file__)
    publish_manifest(result, Path(out)/source.name/'replay_checks.json')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    replay(args.source, args.out)
