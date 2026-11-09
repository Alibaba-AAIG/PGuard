"""SGLang CLI plugin; PGuard modules are imported only when --interrupt_config is given."""

import functools
import json
import os
from pathlib import Path


POLICY_ENV = "PGUARD_INTERRUPT_POLICY"
_HEAD_PATH_ENV = {"stateless": "PGUARD_MODEL_PATH", "seq": "PGUARD_SEQ_STATE_PATH",
                  "cfc": "PGUARD_CFC_STATE_PATH"}


def read_policy(path, model_path=None):
    from pguard.interrupt_policy import load_policy
    from pguard.interrupt_runtime import modeling_variant

    variant = modeling_variant()
    if variant not in _HEAD_PATH_ENV:
        raise ValueError("interrupt_config requires the stateless, seq or cfc variant")
    head = os.getenv(_HEAD_PATH_ENV[variant], "").strip()
    if not head and variant == "stateless" and model_path:
        head = str(Path(model_path) / "guard_model")
    if not head:
        raise ValueError(f"interrupt_config requires {_HEAD_PATH_ENV[variant]}")
    head_path = Path(head).expanduser().resolve()
    config_path = str(head_path / "config.json" if head_path.is_dir()
                      else head_path.parent / "config.json")
    with Path(path).expanduser().open(encoding="utf-8") as handle:
        rules = json.load(handle)
    return load_policy(rules, config_path), rules, config_path


def load():
    from sglang.srt.server_args import ServerArgs

    if not getattr(ServerArgs.add_cli_args, "_pguard_interrupt_cli", False):
        original_add = ServerArgs.add_cli_args
        original_from = ServerArgs.from_cli_args.__func__

        @functools.wraps(original_add)
        def add_cli_args(parser):
            original_add(parser)
            parser.add_argument("--interrupt_config", type=str, default=None,
                                help="Path to the PGuard stage-scoped interrupt policy JSON")

        @functools.wraps(original_from)
        def from_cli_args(cls, args):
            path = getattr(args, "interrupt_config", None)
            # Rebuild from the CLI on every launch so a previous policy is never inherited.
            os.environ.pop(POLICY_ENV, None)
            if path is None:
                return original_from(cls, args)
            from pguard.interrupt_runtime import validate_runtime

            policy, rules, config_path = read_policy(path, getattr(args, "model_path", None))
            server_args = original_from(cls, args)
            validate_runtime(server_args, policy)
            if policy.enabled:
                # Ship the validated rules, not the path, so schedulers never reread the file.
                os.environ[POLICY_ENV] = json.dumps({"rules": rules, "head_config": config_path})
            return server_args

        add_cli_args._pguard_interrupt_cli = True
        ServerArgs.add_cli_args = staticmethod(add_cli_args)
        ServerArgs.from_cli_args = classmethod(from_cli_args)
